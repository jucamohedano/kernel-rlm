"""Private author fixture and CPU eager oracle. Never loads upstream/model source."""

import ctypes as c
import sys

from device_abi import GROUPS, INPUTS, K, M, N, digest

TASK_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
TASK_SHA256 = "461335ed064d25b193afd8cee6645eb12b37540ce15baa5170f3d61b765d2a90"
PREDECESSOR_RECEIPT_SHA256 = "164161c574c98fae8488c03e81b5b108eeee29b3f47c696f0d225587ed668d3b"
EPS, RTOL, ATOL = 1e-5, 1e-4, 1e-4
STATE_HASHES = {
    "upstream": (
        "010b49659f0424633d8c6770053aaa0870b2b3ea1eb40c3aa7372dfba966ff9a",
        "6fbdae69d50489e933169a16fefb85b70195bfa03f458eaec784e26040315c9f",
        "46ac448298521650c9dfcc1d8dad9f941244e7bb071dc84c6841750d864015a9",
        "c35020473aed1b4642cd726cad727b63fff2824ad68cedd7ffb73c7cbd890479",
    ),
    "diagnostic": (
        "1bd216d1ae935996ebb31ef2f833a25d472cb0c23482eff7e8260811bd9f4c75",
        "7771747b17169bd36fb5ed7fc70c3c4d2783577bb7d26eb5ad9872d82ce35d9c",
        "09a350ac3b94745e8ee47aa1572abe7f75e56ad1711174bac4d33fa49ae209ba",
        "afe58fe81a1a44172ca42d3d7f937abff82860c367f746e7583615bafb9784ef",
    ),
}
X_HASHES = {
    108: "51b8860c43a12aee395099352140c32e352478003b5d763e5100cf7c43656a95",
    810: "46aa274fae25b5cb188146274921d1a90fdb46cd3bb548881559ec431e393095",
    17: "1756ebf329c45a09d2783d4c99f8a1dbbe94284d7799658d71e145572b11222a",
    31: "baa649faf41109aaff9dd821dcf8c6388b590b229bd96f458f9f781e1fc92994",
}


def fp32_bytes(tensor, torch) -> bytes:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError("expected contiguous CPU FP32 snapshot")
    return c.string_at(tensor.data_ptr(), 4 * tensor.numel())


def checked_bytes(tensor, expected_hash: str, torch) -> bytes:
    raw = fp32_bytes(tensor, torch)
    if digest(raw) != expected_hash:
        raise ValueError("fixture bytes differ from pinned receipt; no author launch")
    return raw


def fixtures(label: str, control: str):
    import torch
    import torch.nn.functional as F

    if sys.byteorder != "little":
        raise ValueError("expected little-endian host")
    threads = min(8, torch.get_num_threads())
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = False
    diagnostic = label == "diagnostic"
    seeds = (17, 31) if diagnostic else (108, 810)
    if control not in ("positive", "stale_producer"):
        seeds = seeds[:1]

    # Reproduce constructors only; no upstream Python is imported, compiled or executed.
    with torch.inference_mode(), torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(42)
        linear = torch.nn.Linear(K, N, dtype=torch.float32)
        norm = torch.nn.GroupNorm(GROUPS, N, dtype=torch.float32)
        w, b, gamma, beta = linear.weight, linear.bias, norm.weight, norm.bias
        if diagnostic:
            ch = torch.arange(N)
            group = ch // (N // GROUPS)
            w.zero_()
            w[ch, ch % K] = (0.3 + (ch % 7) * 0.01875) * torch.where(
                group == 0, 0.0, torch.where(group == 1, 0.002, 1.0)
            )
            b.copy_(torch.where(group < 2, (group + 1) * 0.125, group * 0.25 + (ch % 5) * 0.03))
            gamma.copy_(0.8 + (ch % 11) * 0.07)
            beta.copy_((ch % 9 - 4) * 0.13 + 3.5 * (ch % 13 == 8) - 3.5 * (ch % 17 == 5))
        state = {
            spec.name: checked_bytes(value, hash_, torch)
            for spec, value, hash_ in zip(
                INPUTS[1:], (w, b, gamma, beta), STATE_HASHES[label], strict=True
            )
        }
        snapshots, refs, coverage = [], [], []
        for seed in seeds:
            torch.default_generator.manual_seed(seed)
            if diagnostic:
                x = torch.randn(M, K) * 0.35 + 0.05 * ((torch.arange(M) % 5) - 2)[:, None]
            else:
                x = torch.rand(M, K)
            snapshots.append(checked_bytes(x, X_HASHES[seed], torch))
            y = F.hardtanh(F.group_norm(F.linear(x, w, b), GROUPS, gamma, beta, EPS), -2.0, 2.0)
            if (
                tuple(y.shape) != (M, N)
                or y.dtype != torch.float32
                or not y.is_contiguous()
                or not torch.isfinite(y).all().item()
            ):
                raise ValueError("malformed independent reference")
            counts = {
                "interior_margin_0_25": int((y.abs() <= 1.75).sum().item()),
                "lower_clamp": int((y == -2.0).sum().item()),
                "upper_clamp": int((y == 2.0).sum().item()),
            }
            if diagnostic and (
                counts["interior_margin_0_25"] < M * N // 4
                or min(counts["lower_clamp"], counts["upper_clamp"]) == 0
            ):
                raise ValueError("diagnostic oracle lacks mixed coverage")
            coverage.append(counts)
            refs.append(y)
    return (
        {"X": snapshots[0], **state},
        snapshots[1] if len(snapshots) > 1 else None,
        refs,
        {
            "oracle_version": "local-trusted-pytorch-cpu-f32-functional/1",
            "pytorch": str(torch.__version__),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn": torch.backends.mkldnn.enabled,
            "threads": threads,
            "fixture": label,
            "state_seed": 42,
            "input_seeds": seeds,
            "reference_sha256": [digest(fp32_bytes(y, torch)) for y in refs],
            "pinned_task_commit": TASK_COMMIT,
            "pinned_task_sha256": TASK_SHA256,
            "predecessor_receipt_sha256": PREDECESSOR_RECEIPT_SHA256,
            "coverage": coverage,
            "atol": ATOL,
            "rtol": RTOL,
        },
    )


def compare(scratch: bytes, actual: bytes, expected) -> dict:
    import torch

    if len(actual) != len(scratch) or len(actual) != expected.numel() * 4:
        raise ValueError("wrong-sized scratch/output bytes")
    # frombuffer borrows read-only storage; no mutation is performed.
    z = torch.frombuffer(scratch, dtype=torch.float32)
    got = torch.frombuffer(actual, dtype=torch.float32).reshape(expected.shape)
    if not torch.isfinite(expected).all().item():
        raise ValueError("nonfinite reference")
    report = {
        "scratch_finite": bool(torch.isfinite(z).all().item()),
        "scratch_all_nan": bool(torch.isnan(z).all().item()),
        "output_finite": bool(torch.isfinite(got).all().item()),
        "output_all_nan": bool(torch.isnan(got).all().item()),
        "passed": False,
    }
    if report["output_finite"]:
        bad = ~torch.isclose(expected, got, atol=ATOL, rtol=RTOL)
        report.update(
            passed=not bad.any().item(),
            mismatched=int(bad.sum().item()),
            interior_mismatched=int(
                (bad & (expected.abs() <= 1.75) & (got.abs() <= 1.75)).sum().item()
            ),
            max_abs=float((expected.double() - got.double()).abs().max().item()),
        )
    return report
