#!/usr/bin/env python3
"""Opt-in author control, not a FlashInfer-Bench runner or a candidate evaluator."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path

import torch

NAME = "gqa_paged_decode_h16_kv2_d128_ps64"
DEFINITION_SHA256 = "6f05748fd5fa488f593009adbe86e4f260fb176ea69a073951b50b79b894c899"
Q_HEADS, KV_HEADS, DIM, PAGE = 16, 2, 128, 64
ATOL, RTOL = 0.01, 0.01
Outputs = tuple[torch.Tensor, torch.Tensor]
Reference = Callable[..., Outputs]


def emit(stage: str, **fields: object) -> None:
    print(json.dumps({"author_only": True, "stage": stage, **fields}, allow_nan=False), flush=True)


def stage_reference(definition: Path, stage_dir: Path) -> Reference:
    raw = definition.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DEFINITION_SHA256:
        raise ValueError(f"definition SHA-256 mismatch: {digest}; expected {DEFINITION_SHA256}")
    doc = json.loads(raw)
    source = doc["reference"]
    stage_dir.mkdir(exist_ok=False)
    (stage_dir / f"{NAME}.json").write_bytes(raw)
    path = stage_dir / "pinned_reference.py"
    path.write_bytes(source.encode("utf-8"))
    emit("reference_staged", definition_sha256=digest, reference_path=str(path))
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["run"]


@dataclass(frozen=True)
class Case:
    name: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    tail: torch.Tensor
    scale: float

    def args(self) -> tuple:
        return self.q, self.k, self.v, self.indptr, self.indices, self.tail, self.scale

    def on(self, device: torch.device) -> "Case":
        return Case(self.name, *(x.to(device) for x in self.args()[:-1]), self.scale)

    def token_counts(self) -> list[int]:
        return [
            (count - 1) * PAGE + tail if count else 0
            for count, tail in zip(self.indptr.diff().tolist(), self.tail.tolist(), strict=True)
        ]

    def describe(self) -> dict:
        return {
            "case": self.name,
            "B": self.q.shape[0],
            "P": self.k.shape[0],
            "M": self.indices.numel(),
            "tokens_per_row": self.token_counts(),
            "sm_scale": self.scale,
            "input_tensor_bytes": sum(x.numel() * x.element_size() for x in self.args()[:-1]),
        }


def int32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def analytic_probe() -> tuple[Case, Outputs]:
    q = torch.zeros((1, Q_HEADS, DIM), dtype=torch.bfloat16)
    k = torch.full((2, PAGE, KV_HEADS, DIM), 16.0, dtype=q.dtype)
    v = torch.full_like(k, -16.0)
    q[:, :, 0] = 2
    k[1, :2, :, 0] = torch.tensor([[0, -1], [1, 0]], dtype=q.dtype)
    v[1, :2].zero_()
    v[1, :2, :, 7] = torch.tensor([[2, -3], [-2, 1]], dtype=q.dtype)
    probe = Case("analytic_2_tokens", q, k, v, int32([0, 1]), int32([1]), int32([2]), 0.25)
    # Natural logits [0, 1/2] and [-1/2, 0]. Both groups have weights 1:r.
    r = math.exp(0.5)
    out = torch.zeros_like(q)
    out[:, :8, 7] = (2 - 2 * r) / (1 + r)
    out[:, 8:, 7] = (-3 + r) / (1 + r)
    lse = torch.empty((1, Q_HEADS), dtype=torch.float32)
    lse[:, :8] = math.log2(1 + r)
    lse[:, 8:] = math.log2(1 + r) - 0.5 / math.log(2)
    return probe, (out, lse)


def irregular() -> Case:
    rng = torch.Generator(device="cpu").manual_seed(411)
    q = torch.randn((5, Q_HEADS, DIM), generator=rng).to(torch.bfloat16)
    k = torch.randn((8, PAGE, KV_HEADS, DIM), generator=rng).to(torch.bfloat16)
    v = torch.randn(k.shape, generator=rng).to(torch.bfloat16)
    # Only the final pages unique to these rows are poisoned beyond their used slots.
    for page, used in [(5, 1), (1, 63), (3, 1), (2, 25), (0, 0)]:
        k[page, used:].fill_(16)
        v[page, used:].fill_(-16)
    return Case(
        "irregular_nonempty",
        q,
        k,
        v,
        int32([0, 1, 2, 3, 5, 8]),
        int32([5, 1, 6, 6, 3, 7, 7, 2]),
        int32([1, 63, 64, 1, 25]),
        0.125,
    )


def mixed_empty(case: Case) -> Case:
    # Replace the first row by a zero-page row; leave the other four populated.
    return replace(
        case,
        name="mixed_empty_sentinel",
        indptr=int32([0, 0, 1, 2, 4, 7]),
        indices=case.indices[1:].contiguous(),
        tail=int32([0, 63, 64, 1, 25]),
    )


def timing_cases() -> tuple[Case, Case]:
    rng = torch.Generator(device="cpu").manual_seed(1228)
    counts = [9, 17, 33, 65] * 4
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    pages = offsets[-1] + 16
    q = torch.randn((len(counts), Q_HEADS, DIM), generator=rng).to(torch.bfloat16)
    k = torch.randn((pages, PAGE, KV_HEADS, DIM), generator=rng).to(torch.bfloat16)
    v = torch.randn(k.shape, generator=rng).to(torch.bfloat16)
    case = Case(
        "timing_a",
        q,
        k,
        v,
        int32(offsets),
        torch.randperm(pages, generator=rng)[: offsets[-1]].int(),
        int32([64, 29, 3, 47] * 4),
        1 / math.sqrt(DIM),
    )
    # Same device-resident Q/K/V and sizes; a different rotation of page IDs and tails.
    alt = replace(
        case,
        name="timing_b",
        indices=(case.indices + 13) % pages,
        tail=case.tail.roll(1),
    )
    return case, alt


def oracle(case: Case) -> Outputs:
    """FP64 on CPU: walk logical token addresses, group heads via slicing, use exp2."""
    o = torch.zeros_like(case.q)
    lse = torch.full((case.q.shape[0], Q_HEADS), -math.inf, dtype=torch.float32)
    for b, count in enumerate(case.token_counts()):
        if count == 0:
            continue
        logical = torch.arange(count)
        pages = case.indices[case.indptr[b] + logical // PAGE].long()
        slots = logical % PAGE
        for kv in range(KV_HEADS):
            heads = slice(kv * 8, (kv + 1) * 8)
            q = case.q[b, heads].double()
            k = case.k[pages, slots, kv].double()
            v = case.v[pages, slots, kv].double()
            u = (q @ k.T) * (case.scale * math.log2(math.e))
            shift = u.amax(dim=1, keepdim=True)
            weights = torch.exp2(u - shift)
            mass = weights.sum(dim=1, keepdim=True)
            o[b, heads] = ((weights @ v) / mass).to(torch.bfloat16)
            lse[b, heads] = (shift + torch.log2(mass)).flatten().float()
    return o, lse


def compare(got: Outputs, want: Outputs, counts: list[int]) -> dict:
    """All elements pass the pinned nonempty tolerance; empties use exact sentinels."""
    if not isinstance(got, tuple) or len(got) != 2:
        raise ValueError("expected (output, lse)")
    empty = torch.tensor([t == 0 for t in counts], dtype=torch.bool)
    report = {"empty_rows": int(empty.sum())}
    for name, actual, target in zip(("output", "lse"), got, want, strict=True):
        if actual.shape != target.shape or actual.dtype != target.dtype:
            raise ValueError(f"{name} shape/dtype mismatch: {actual.shape}, {actual.dtype}")
        actual, target = actual.cpu(), target.cpu()
        if empty.any():
            for tensor in (actual, target):
                valid = tensor[empty] == 0 if name == "output" else torch.isneginf(tensor[empty])
                if not valid.all():
                    raise ValueError(f"{name} bad empty sentinel")
        x, y = actual[~empty].float(), target[~empty].float()
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError(f"{name} nonfinite on nonempty row")
        diff = (x - y).abs()
        rel = diff / (y.abs() + 1e-8)
        fails = int(((diff > ATOL) & (rel > RTOL)).sum())
        report[name] = {
            "numel": x.numel(),
            "mismatches": fails,
            "max_abs": float(diff.max()) if x.numel() else None,
            "max_rel": float(rel.max()) if x.numel() else None,
        }
    return report


def check(case: Case, got: Outputs, want: Outputs, implementation: str) -> None:
    report = compare(got, want, case.token_counts())
    emit("numerics", **case.describe(), implementation=implementation, **report)
    if any(report[name]["mismatches"] for name in ("output", "lse")):
        raise ValueError(f"{implementation} numeric gate failed: {case.name}")


def controls(probe: Case, got: Outputs, want: Outputs) -> dict:
    # For this probe, Q is identical for every head, so an interleaved head
    # mapping is exactly each expected head group's result routed to h % 2.
    # Remap a known-correct output, changing both O and LSE for those heads.
    heads = torch.tensor([0, 8] * 8, device=got[0].device)
    wrong_head = compare((got[0][:, heads], got[1][:, heads]), want, probe.token_counts())
    wrong_log = compare((got[0], got[1] * math.log(2)), want, probe.token_counts())
    if not (
        wrong_head["output"]["mismatches"]
        and wrong_head["lse"]["mismatches"]
        and not wrong_log["output"]["mismatches"]
        and wrong_log["lse"]["mismatches"]
    ):
        raise ValueError("negative controls failed to discriminate O and base-2 LSE")
    return {"interleaved_kv_head": wrong_head, "natural_log_lse": wrong_log}


def reference_gate(reference: Reference, device: torch.device) -> list[tuple[Case, Outputs]]:
    probe, exact = analytic_probe()
    ragged = irregular()
    empty = mixed_empty(ragged)
    fixtures = [(probe, exact), (ragged, oracle(ragged)), (empty, oracle(empty))]
    for case, want in fixtures:
        got = reference(*case.on(device).args())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        check(case, got, want, "pinned_eager_reference")
        if case is probe:
            emit("controls_rejected", **controls(probe, got, want))
    emit("reference_pass", device=str(device), timing=False, stock_evaluator_used=False)
    return fixtures[:2]


class NativeDecode:
    def __init__(self, wrapper_class, backend, workspace, case):
        self.wrapper = wrapper_class(
            workspace,
            kv_layout="NHD",
            use_cuda_graph=False,
            use_tensor_cores=backend == "fa2-tensor",
            backend="fa2",
        )
        self.out = torch.empty_like(case.q)
        self.lse = torch.empty(case.q.shape[:2], dtype=torch.float32, device=case.q.device)

    def plan(self, case: Case) -> None:
        self.wrapper.plan(
            case.indptr,
            case.indices,
            case.tail,
            Q_HEADS,
            KV_HEADS,
            DIM,
            PAGE,
            pos_encoding_mode="NONE",
            window_left=-1,
            logits_soft_cap=0.0,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=case.scale,
        )

    def run(self, case: Case) -> Outputs:
        return self.wrapper.run(
            case.q,
            (case.k, case.v),
            out=self.out,
            lse=self.lse,
            return_lse=True,
        )

    def check(self, case: Case, host: Case, want: Outputs, stream, implementation: str) -> None:
        # A correct native call must write both advertised destinations. Never time this fill.
        self.out.fill_(math.nan)
        self.lse.fill_(math.nan)
        stream.synchronize()
        emit("native_plan_start", case=case.name)
        start = time.perf_counter()
        self.plan(case)
        stream.synchronize()
        module = getattr(self.wrapper, "_cached_module", None)
        emit(
            "native_plan_complete",
            case=case.name,
            first_plan_wall_ms=(time.perf_counter() - start) * 1e3,
            cached_module_type=type(module).__name__ if module is not None else None,
            includes_any_first_use_jit_or_load=True,
        )
        got = self.run(case)
        emit("native_launch_returned", case=case.name)
        stream.synchronize()
        emit("native_stream_complete", case=case.name)
        if got[0].data_ptr() != self.out.data_ptr() or got[1].data_ptr() != self.lse.data_ptr():
            raise ValueError("native wrapper did not return the caller-owned destinations")
        check(host, (self.out, self.lse), want, implementation)


def sample_distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not all(math.isfinite(v) and v > 0 for v in ordered):
        raise ValueError("invalid timing samples")
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "max_ms": ordered[-1],
        "samples_ms": values,
    }


def time_native(decoder: NativeDecode, cases: tuple[Case, Case], stream, warmup, repeats) -> None:
    first, _ = cases
    decoder.plan(first)
    stream.synchronize()
    for _ in range(warmup):
        decoder.run(first)
    stream.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    # CUDA events are lazily initialized. Initialize outside the measured interval.
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        end.record(stream)
    stream.synchronize()
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        decoder.run(first)
        end.record(stream)
    stream.synchronize()
    emit(
        "native_timing",
        scope="fixed_table_run_stream_events",
        cases=[first.name],
        warmup=warmup,
        repeats=repeats,
        plan_reused=True,
        cuda_graph=False,
        cache_mode="warm_no_flush",
        **sample_distribution([a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]),
    )

    for i in range(warmup):
        case = cases[i % 2]
        decoder.plan(case)
        decoder.run(case)
        stream.synchronize()
    samples = []
    for i in range(repeats):
        case = cases[(warmup + i) % 2]
        start = time.perf_counter()
        decoder.plan(case)
        decoder.run(case)
        stream.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    emit(
        "native_timing",
        scope="changing_table_plan_run_sync_host_wall",
        cases=[case.name for case in cases],
        warmup=warmup,
        repeats=repeats,
        plan_reused=False,
        cuda_graph=False,
        cache_mode="warm_no_flush",
        **sample_distribution(samples),
    )


def native_gate(
    reference: Reference, device: torch.device, backend, workspace_mib, warmup, repeats
):
    emit("native_import_start", backend=backend)
    # Test the already installed native runtime. Import failure is fatal, not an eager fallback.
    import flashinfer.decode as decode

    emit(
        "native_import_complete",
        distribution="flashinfer-python",
        version=metadata.version("flashinfer-python"),
        module_path=decode.__file__,
        decode_source_sha256=hashlib.sha256(Path(decode.__file__).read_bytes()).hexdigest(),
        backend=backend,
        api_backend="fa2",
        use_tensor_cores=backend == "fa2-tensor",
    )
    fixtures = reference_gate(reference, device)
    timing_a, timing_b = timing_cases()
    device_a = timing_a.on(device)
    device_b = replace(
        device_a,
        name=timing_b.name,
        indices=timing_b.indices.to(device),
        tail=timing_b.tail.to(device),
    )
    stream = torch.cuda.current_stream(device)
    workspace = torch.empty(workspace_mib * 2**20, dtype=torch.uint8, device=device)
    implementation = f"native_flashinfer_{backend}"

    def make_decoder(case: Case) -> NativeDecode:
        return NativeDecode(decode.BatchDecodeWithPagedKVCacheWrapper, backend, workspace, case)

    for host, want in fixtures:
        case = host.on(device)
        make_decoder(case).check(case, host, want, stream, implementation)
    del case
    stream.synchronize()
    # Report torch-tracked memory separately from the known caller allocations. This is
    # not total GPU or host workspace: an installed extension can allocate outside torch.
    alloc_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    decoder = make_decoder(device_a)
    for case, host in [(device_a, timing_a), (device_b, timing_b)]:
        want = oracle(host)
        check(host, reference(*case.args()), want, "pinned_eager_reference_timing_fixture")
        decoder.check(case, host, want, stream, implementation)
    output_bytes = sum(t.numel() * t.element_size() for t in (decoder.out, decoder.lse))
    # Exclude eager/independent correctness work from torch peak in the timing stage.
    stream.synchronize()
    live_after_checks = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    time_native(decoder, (device_a, device_b), stream, warmup, repeats)
    emit(
        "native_resources",
        caller_workspace_bytes=workspace.numel(),
        caller_output_bytes=output_bytes,
        input_tensor_bytes_a=timing_a.describe()["input_tensor_bytes"],
        additional_metadata_bytes_b=sum(
            t.numel() * t.element_size() for t in (device_b.indices, device_b.tail)
        ),
        torch_live_delta_after_checks_includes_outputs_and_internal_buffers=live_after_checks
        - alloc_before,
        torch_timed_peak_extra_bytes=torch.cuda.max_memory_allocated(device) - live_after_checks,
        host_and_non_torch_allocations_measured=False,
    )
    emit(
        "native_baseline_pass",
        implementation=implementation,
        scope="tested_nonempty_synthetic_only",
        empty_native_qualification=False,
        candidate_executed=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-only", action="store_true", required=True)
    parser.add_argument(
        "--definition", type=Path, required=True, help="SHA-256 pinned definition JSON"
    )
    parser.add_argument(
        "--stage-dir", type=Path, required=True, help="new directory under approved scratch"
    )
    parser.add_argument(
        "--mode", choices=("cpu-reference", "sm120-reference", "sm120-native"), required=True
    )
    parser.add_argument("--backend", choices=("cuda-core", "fa2-tensor"))
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if (args.backend is None) == (args.mode == "sm120-native"):
        parser.error("choose exactly one explicit --backend for sm120-native only")
    if args.workspace_mib < 1 or args.warmup < 2 or args.repeats < 2:
        parser.error("workspace must be positive; warmup and repeats must be at least two")

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    emit(
        "start",
        mode=args.mode,
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_cuda=torch.version.cuda,
    )
    reference = stage_reference(args.definition, args.stage_dir)
    with torch.no_grad():
        if args.mode == "cpu-reference":
            reference_gate(reference, torch.device("cpu"))
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; cannot qualify sm_120")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) != (12, 0):
            raise RuntimeError(
                f"expected capability (12,0); got ({props.major},{props.minor}) on {props.name}"
            )
        emit(
            "device",
            name=props.name,
            capability=[props.major, props.minor],
            memory_bytes=props.total_memory,
            gpu_uuid=str(getattr(props, "uuid", "unknown")),
            matmul_precision=torch.get_float32_matmul_precision(),
        )
        with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
            if args.mode == "sm120-reference":
                reference_gate(reference, device)
            else:
                native_gate(
                    reference, device, args.backend, args.workspace_mib, args.warmup, args.repeats
                )


if __name__ == "__main__":
    main()
