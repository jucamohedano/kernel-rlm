"""Pinned, synthetic-only SFT bundle and shared row-local loss."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Member:
    input_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class Group:
    trajectory_id: str
    problem_group: str
    members: tuple[Member, ...]


@dataclass(frozen=True)
class Bundle:
    model_revision: str
    tokenizer_revision: str
    pad_id: int
    member_cap: int
    padded_slots_cap: int
    groups: tuple[Group, ...]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _read(root, name, digest):
    path = root / name
    _require(path.is_file() and not path.is_symlink(), "expected a local bundle file")
    data = path.read_bytes()
    _require(type(digest) is str and _sha(data) == digest, f"pin mismatch: {name}")
    return data


def _limit(n):
    return type(n) is int and n > 0


def _check_group(group, member_cap, padded_slots_cap, vocab_size=None):
    _require(
        group.members and _limit(member_cap) and _limit(padded_slots_cap), "invalid group or caps"
    )
    _require(max(len(m.input_ids) for m in group.members) <= member_cap, "member over cap")
    _require(
        len(group.members) * max(len(m.input_ids) for m in group.members) <= padded_slots_cap,
        "entire trajectory exceeds padded slot cap",
    )
    count = 0
    for member in group.members:
        ids, mask = member.input_ids, member.loss_mask
        _require(
            bool(ids) and len(ids) == len(mask) and mask[0] == 0,
            "invalid member mask shape or shift",
        )
        _require(
            all(type(x) is int and 0 <= x and (vocab_size is None or x < vocab_size) for x in ids),
            "invalid token ID",
        )
        _require(all(type(x) is int and x in (0, 1) for x in mask), "non-binary loss mask")
        count += sum(mask)
    _require(count > 0, "empty trajectory loss mask")
    return count


def load_bundle(directory, expected_manifest_sha256, *, allow_synthetic=False):
    """Load exact, locally pinned synthetic rows. Real artifacts cannot opt in."""
    root = Path(directory)
    manifest = json.loads(
        _read(root, "manifest.json", expected_manifest_sha256), object_pairs_hook=_object
    )
    _require(
        isinstance(manifest, dict)
        and manifest.get("scope") == "synthetic_cpu"
        and allow_synthetic is True,
        "production blocked: no rights, split, capacity or backend receipt",
    )
    keys = {
        "schema",
        "scope",
        "model_revision",
        "tokenizer_revision",
        "pad_id",
        "vocab_size",
        "member_cap",
        "padded_slots_cap",
        "rows_sha256",
        "total_groups",
        "total_targets",
    }
    _require(
        set(manifest) == keys and manifest["schema"] == "xs21-grouped-bundle-v1",
        "unexpected manifest schema",
    )
    _require(
        all(
            type(manifest[key]) is str and re.fullmatch("[0-9a-f]{40}", manifest[key])
            for key in ("model_revision", "tokenizer_revision")
        ),
        "invalid synthetic revision pin",
    )
    pad_id, vocab_size, member_cap, slots = (
        manifest[k] for k in ("pad_id", "vocab_size", "member_cap", "padded_slots_cap")
    )
    _require(
        _limit(vocab_size) and type(pad_id) is int and 0 <= pad_id < vocab_size,
        "invalid vocabulary or pad ID",
    )
    rows = _read(root, "groups.jsonl", manifest["rows_sha256"])
    groups, seen, tokens = [], set(), 0
    for raw in rows.splitlines():
        row = json.loads(raw, object_pairs_hook=_object)
        _require(
            isinstance(row, dict) and set(row) == {"trajectory_id", "problem_group", "members"},
            "invalid trajectory schema",
        )
        tid, problem, members = (row[k] for k in ("trajectory_id", "problem_group", "members"))
        _require(
            all(
                type(s) is str and bool(re.fullmatch(r"synthetic-[a-z0-9-]+", s))
                for s in (tid, problem)
            )
            and tid not in seen
            and type(members) is list,
            "invalid or duplicate synthetic identity",
        )
        seen.add(tid)
        parsed = []
        for item in members:
            _require(
                type(item) is dict
                and set(item) == {"input_ids", "loss_mask"}
                and all(type(t) is list for t in item.values()),
                "invalid decision schema",
            )
            parsed.append(Member(tuple(item["input_ids"]), tuple(item["loss_mask"])))
        group = Group(tid, problem, tuple(parsed))
        tokens += _check_group(group, member_cap, slots, vocab_size)
        groups.append(group)
    _require(
        _limit(manifest["total_groups"])
        and _limit(manifest["total_targets"])
        and len(groups) == manifest["total_groups"]
        and tokens == manifest["total_targets"],
        "manifest count differs",
    )
    return Bundle(
        manifest["model_revision"],
        manifest["tokenizer_revision"],
        pad_id,
        member_cap,
        slots,
        tuple(groups),
    )


def collate_group(group, *, pad_id, member_cap, padded_slots_cap):
    _require(type(pad_id) is int and pad_id >= 0, "invalid pad ID")
    _check_group(group, member_cap, padded_slots_cap)
    width = max(len(m.input_ids) for m in group.members)
    ids = torch.full((len(group.members), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    for i, m in enumerate(group.members):
        n = len(m.input_ids)
        ids[i, :n] = torch.tensor(m.input_ids, dtype=torch.long)
        attention[i, :n] = True
        mask = torch.tensor(m.loss_mask, dtype=torch.bool)
        labels[i, :n] = ids[i, :n].masked_fill(~mask, -100)
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def masked_nll_sum(logits, labels):
    """Return causally shifted summed NLL and exact count; caller owns normalization."""
    _require(
        logits.ndim == 3
        and labels.ndim == 2
        and logits.shape[:2] == labels.shape
        and labels.dtype == torch.long
        and logits.device == labels.device,
        "incompatible logits and labels",
    )
    _require(not bool((labels[:, 0] != -100).any()), "no previous token for label at position zero")
    next_labels = labels[:, 1:]
    use = next_labels != -100
    keep = logits[:, :-1][use].float()
    n = use.sum()
    if keep.shape[0] == 0:
        return keep.sum(), n
    return F.cross_entropy(keep, next_labels[use], reduction="sum"), n


def write_synthetic_fixture(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    groups = [
        {
            "trajectory_id": "synthetic-a",
            "problem_group": "synthetic-p-a",
            "members": [
                {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1]},
                {"input_ids": [5, 0, 6], "loss_mask": [0, 0, 0]},
            ],
        },
        {
            "trajectory_id": "synthetic-b",
            "problem_group": "synthetic-p-b",
            "members": [
                {"input_ids": [5, 1, 7], "loss_mask": [0, 0, 1]},
            ],
        },
    ]
    rows = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in groups
    )
    manifest = dict(
        schema="xs21-grouped-bundle-v1",
        scope="synthetic_cpu",
        model_revision="0" * 40,
        tokenizer_revision="1" * 40,
        pad_id=0,
        vocab_size=16,
        member_cap=4,
        padded_slots_cap=8,
        rows_sha256=_sha(rows),
        total_groups=2,
        total_targets=3,
    )
    data = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with (root / "groups.jsonl").open("xb") as f:
        f.write(rows)
    with (root / "manifest.json").open("xb") as f:
        f.write(data)
    return _sha(data)
