#!/usr/bin/env python3
"""Opt-in KernelBench L2/30 reference qualification, not the upstream evaluator."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any

import torch
from native_controls import AuthorControls

SOURCE_SHA256 = "461335ed064d25b193afd8cee6645eb12b37540ce15baa5170f3d61b765d2a90"
SOURCE_COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
FULL = (1024, 8192, 8192, 16)  # M, K, N, G
SMOKE = (8, 32, 64, 4)
ATOL = RTOL = 1e-4
STATE_SEED = 42
UPSTREAM_SEEDS = (108, 810)
DIAGNOSTIC_SEEDS = (17, 31)
EAGER_LABEL = "pinned_pytorch_eager_reference"
TUNED_LABEL = "pytorch_inductor_max_autotune_no_cudagraphs"
STATE_NAMES = ("gemm.weight", "gemm.bias", "group_norm.weight", "group_norm.bias")


def load_pinned(path: Path) -> types.ModuleType:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("upstream SHA-256 mismatch (refusing to execute)")
    module = types.ModuleType("kernelbench_l2_30_pinned")
    module.__file__ = str(path.resolve())
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def make_model(upstream: types.ModuleType, dims: tuple, device: torch.device) -> torch.nn.Module:
    m, k, n, groups = dims
    if dims == FULL:
        if upstream.batch_size != m or upstream.get_init_inputs() != [k, n, groups, -2.0, 2.0]:
            raise ValueError("upstream shape/parameters disagree with contract")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(STATE_SEED)
        model = upstream.Model(k, n, groups, -2.0, 2.0)
    return model.eval().to(device=device, dtype=torch.float32)


def state_tensors(model: torch.nn.Module) -> tuple:
    state = model.state_dict(keep_vars=True)
    if tuple(state) != STATE_NAMES:
        raise ValueError("unexpected reference state")
    return tuple(state[name] for name in STATE_NAMES)


def make_input(
    upstream: types.ModuleType, dims: tuple, seed: int, diagnostic: bool, device: torch.device
) -> torch.Tensor:
    m, k, _, _ = dims
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(seed)
        if diagnostic:
            x = torch.randn(m, k) * 0.35 + 0.05 * ((torch.arange(m) % 5) - 2)[:, None]
        elif dims == FULL:
            (x,) = upstream.get_inputs()
        else:
            x = torch.rand(m, k)
    if x.shape != (m, k) or x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("unexpected reference input layout/dtype")
    return x.to(device=device)


def change_to_diagnostic_state(state: tuple, dims: tuple) -> None:
    """Overwrite the same tensor storage used by eager, oracle and author controls."""
    _, k, n, groups = dims
    w, b, gamma, beta = state
    c = torch.arange(n, device=w.device)
    g = c // (n // groups)
    w.zero_()
    w[c, c % k] = (0.3 + (c % 7) * 0.01875) * torch.where(
        g == 0, 0.0, torch.where(g == 1, 0.002, 1.0)
    )
    b.copy_(torch.where(g < 2, (g + 1) * 0.125, g * 0.25 + (c % 5) * 0.03))
    gamma.copy_(0.8 + (c % 11) * 0.07)
    beta.copy_((c % 9 - 4) * 0.13 + 3.5 * (c % 13 == 8) - 3.5 * (c % 17 == 5))


def equation_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # FP64 dot against the actual FP32 operands; cast at the Linear boundary.
    return (torch.mm(x.double(), w.double().T) + b.double()).float()


def equation_finish(
    z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, groups: int, fault: str | None = None
) -> torch.Tensor:
    m, n = z.shape
    width = n // groups
    blocks = z.double().reshape(m, groups, width)
    if fault == "interleaved_groups":
        blocks = z.double().reshape(m, width, groups).transpose(1, 2)
    centered = blocks - blocks.mean(dim=-1, keepdim=True)
    divisor = width - (fault == "sample_variance")
    var = (centered * centered).sum(dim=-1, keepdim=True) / divisor
    values = centered / (var + 1e-5).sqrt()
    if fault == "interleaved_groups":
        values = values.transpose(1, 2)
    return (values.reshape(m, n) * gamma.double() + beta.double()).float().clamp(-2.0, 2.0)


def fp32_sha256(tensor: torch.Tensor) -> str:
    # Hash raw contiguous bytes of the actual FP32 buffer, without a NumPy dependency.
    t = tensor.detach().cpu()
    if t.dtype != torch.float32 or not t.is_contiguous():
        raise ValueError("unexpected tensor layout for FP32 byte hash")
    data = (ctypes.c_char * (t.numel() * t.element_size())).from_address(t.data_ptr())
    return hashlib.sha256(data).hexdigest()


def counts(y: torch.Tensor) -> dict:
    return {
        "elements": y.numel(),
        "interior_margin_0_25": (y.abs() <= 1.75).count_nonzero().item(),
        "lower_clamp": (y == -2).count_nonzero().item(),
        "upper_clamp": (y == 2).count_nonzero().item(),
    }


def checked(actual: torch.Tensor, ref: torch.Tensor) -> dict:
    if (
        actual.shape != ref.shape
        or actual.dtype != ref.dtype
        or actual.device != ref.device
        or not actual.is_contiguous()
    ):
        return {
            "passed": False,
            "reason": "shape_dtype_device_or_contiguity",
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "device": str(actual.device),
        }
    if not torch.isfinite(actual).all().item():
        return {"passed": False, "reason": "nonfinite"}
    # Same argument order as the pinned upstream evaluator; compare every element.
    bad = ~torch.isclose(ref, actual, atol=ATOL, rtol=RTOL)
    return {
        "passed": not bad.any().item(),
        "mismatched": bad.count_nonzero().item(),
        "interior_mismatched": (bad & (ref.abs() <= 1.75) & (actual.abs() <= 1.75))
        .count_nonzero()
        .item(),
        "max_abs": (actual - ref).abs().max().item(),
    }


def require_pass(report: dict, name: str) -> None:
    if not report["passed"]:
        raise RuntimeError(f"{name}: correctness failed: {report}")


def require_negative_rejected(report: dict, name: str) -> None:
    if report["passed"] or not report.get("interior_mismatched"):
        raise RuntimeError(f"{name}: expected finite failure on interior outputs, got {report}")


def validate_unchanged(live: tuple, snapshots: tuple) -> None:
    if not all(torch.equal(a, b) for a, b in zip(live, snapshots, strict=True)):
        raise RuntimeError("forward mutated input or explicit state")


def validate_call(
    fn: Any, x: torch.Tensor, state: tuple, ref: torch.Tensor, stream: Any, report: dict, name: str
) -> None:
    live = (x, *state)
    snapshots = tuple(t.clone() for t in live)
    for attempt in ("first", "repeat"):
        stage = report[attempt] = {"call_returned": False, "stream_completed": False}
        actual = fn(x)
        stage["call_returned"] = True
        if stream is not None:
            stream.synchronize()
            stage["stream_completed"] = True
        stage["comparison"] = checked(actual, ref)
        require_pass(stage["comparison"], f"{name}/{attempt}")
        validate_unchanged(live, snapshots)
        stage["inputs_and_state_unchanged"] = True


def validate_native(
    native: AuthorControls,
    x: torch.Tensor,
    previous: torch.Tensor,
    state: tuple,
    ref: torch.Tensor,
    stream: Any,
    report: dict,
    run_negatives: bool,
) -> None:
    live = (x, previous, *state)
    snapshots = tuple(t.clone() for t in live)
    w, b, gamma, beta = state
    scratch = torch.full_like(ref, float("nan"))

    def run_case(name: str, produced_with: torch.Tensor, fault: int, want_good: bool) -> None:
        stage = report[name] = {
            "producer": "not_enqueued",
            "epilogue": "not_enqueued",
            "stream_completed": False,
        }
        y = torch.full_like(ref, float("nan"))
        native.produce(produced_with, w, b, scratch, stream)
        stage["producer"] = (
            "current_enqueued" if produced_with is x else "stale_enqueued_no_current_producer"
        )
        native.finish(scratch, gamma, beta, y, stream, fault=fault)
        stage["epilogue"] = "enqueued"
        stream.synchronize()
        stage["stream_completed"] = True
        stage["comparison"] = checked(y, ref)
        if want_good:
            require_pass(stage["comparison"], name)
        else:
            require_negative_rejected(stage["comparison"], name)
        validate_unchanged(live, snapshots)
        stage["inputs_and_state_unchanged"] = True

    run_case("positive", x, 0, True)
    scratch.fill_(float("nan"))
    run_case("positive_repeat", x, 0, True)
    if run_negatives:
        run_case("interleaved_groups", x, 1, False)
        run_case("sample_variance", x, 2, False)
        run_case("skipped_current_producer_stale_buffer", previous, 0, False)


def time_pair(fns: dict, x: torch.Tensor, stream: Any, samples: int, calls: int) -> dict:
    values: dict[str, list[float]] = {name: [] for name in fns}
    for fn in fns.values():
        for _ in range(10):
            fn(x)
    stream.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(samples):
        order = list(fns) if i % 2 == 0 else list(reversed(fns))
        for name in order:
            begin.record(stream)
            for _ in range(calls):
                fns[name](x)
            end.record(stream)
            end.synchronize()
            sample = begin.elapsed_time(end) / calls
            if not math.isfinite(sample) or sample <= 0.0:
                raise RuntimeError("invalid CUDA-event timing sample")
            values[name].append(sample)
    return {
        "unit": "ms_per_full_forward",
        "calls_per_event_sample": calls,
        "samples": samples,
        "warmup_calls_per_backend": 10,
        "stream": "same_harness_nondefault_stream",
        "includes": "all device work of Linear+bias, GroupNorm affine, Hardtanh",
        "excludes": "compilation, preparation/uploads, validation and output serialization",
        "backends": {
            name: {
                "raw_sample_ms": v,
                "median_ms": statistics.median(v),
                "min_ms": min(v),
                "max_ms": max(v),
            }
            for name, v in values.items()
        },
    }


def phase(receipt: dict, name: str) -> None:
    receipt["phase"] = name
    print(f"kb30: {name}", file=sys.stderr, flush=True)


def build_tuned(model: torch.nn.Module, report: dict) -> Any:
    import triton
    from torch._dynamo import config as dynamo_config
    from torch._inductor import compile as inductor_compile
    from torch._inductor import list_mode_options

    dynamo_config.suppress_errors = False
    options = {**list_mode_options("max-autotune-no-cudagraphs"), "triton.cudagraphs": False}
    if not options.get("max_autotune"):
        raise RuntimeError("Inductor max-autotune mode did not enable autotuning")
    report.update(inductor_options=options, compiled_graphs=0, triton=triton.__version__)

    def tracked_inductor(graph: Any, example_inputs: list) -> Any:
        result = inductor_compile(graph, example_inputs, options=options)
        report["compiled_graphs"] += 1
        return result

    return torch.compile(model, backend=tracked_inductor, fullgraph=True, dynamic=False)


def run_cases(
    args: argparse.Namespace,
    receipt: dict,
    upstream: types.ModuleType,
    device: torch.device,
    stream: Any = None,
    native: AuthorControls | None = None,
) -> None:
    dims = SMOKE if args.mode == "cpu-smoke" else FULL
    receipt["dimensions_m_k_n_g"] = dims
    model = make_model(upstream, dims, device)
    state = state_tensors(model)
    _, _, _, groups = dims
    timed_x, timed_ref = None, None
    compiled = None
    if args.mode == "gpu-full":
        receipt["tuned"] = {
            "build_and_verify": "started",
            "backend": "inductor",
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": True,
            "dynamic": False,
            "individual_gemm_provider": "not_attributed",
        }
        compiled = build_tuned(model, receipt["tuned"])

    receipt["state_fp32_sha256"] = {}
    receipt["cases"] = {}
    for diagnostic, seeds in ((False, UPSTREAM_SEEDS), (True, DIAGNOSTIC_SEEDS)):
        if diagnostic:
            change_to_diagnostic_state(state, dims)
        state_label = "diagnostic" if diagnostic else "upstream"
        receipt["state_fp32_sha256"][state_label] = {
            key: fp32_sha256(tensor) for key, tensor in zip(STATE_NAMES, state, strict=True)
        }
        for i, seed in enumerate(seeds):
            name = f"{'diagnostic' if diagnostic else 'upstream'}_{seed}"
            phase(receipt, f"{name}: eager and equation")
            case = receipt["cases"][name] = {
                "input_seed": seed,
                "state": "diagnostic" if diagnostic else "upstream",
            }
            x = make_input(upstream, dims, seed, diagnostic, device)
            case["input_fp32_sha256"] = fp32_sha256(x)
            z = equation_linear(x, state[0], state[1])
            ref = equation_finish(z, state[2], state[3], groups)
            case["eager"] = {}
            validate_call(model, x, state, ref, stream, case["eager"], EAGER_LABEL)
            eager = model(x)
            case["coverage"] = coverage = counts(eager)
            if diagnostic and (
                coverage["interior_margin_0_25"] < eager.numel() // 4
                or min(coverage["lower_clamp"], coverage["upper_clamp"]) == 0
            ):
                raise RuntimeError("diagnostic case failed interior/both-bounds coverage")
            if compiled is not None:
                phase(receipt, f"{name}: tuned build/run/check")
                case["tuned"] = {}
                validate_call(compiled, x, state, eager, stream, case["tuned"], TUNED_LABEL)
                if receipt["tuned"]["compiled_graphs"] == 0:
                    raise RuntimeError(
                        "no full graph reached Inductor; refusing eager-as-tuned timing"
                    )
                if not diagnostic and i == 0:
                    timed_x, timed_ref = x, eager
            if not diagnostic and native is not None and i == 0:
                phase(receipt, f"{name}: native positive")
                case["native"] = {}
                validate_native(native, x, x, state, eager, stream, case["native"], False)
            if diagnostic and i == 0:
                phase(receipt, f"{name}: negative controls")
                previous = make_input(upstream, dims, seeds[1], True, device)
                case["stale_input_fp32_sha256"] = fp32_sha256(previous)
                earlier = equation_linear(previous, state[0], state[1])
                case["equation_negatives"] = {}
                for fault in (
                    "interleaved_groups",
                    "sample_variance",
                    "skipped_current_producer_stale_buffer",
                ):
                    bad = equation_finish(
                        earlier if fault.startswith("skipped") else z,
                        state[2],
                        state[3],
                        groups,
                        fault=fault,
                    )
                    case["equation_negatives"][fault] = checked(bad, eager)
                    require_negative_rejected(case["equation_negatives"][fault], fault)
                if native is not None:
                    case["native"] = {}
                    validate_native(native, x, previous, state, eager, stream, case["native"], True)

    if compiled is not None:
        assert stream is not None and timed_x is not None and timed_ref is not None
        # Restore canonical state in-place without changing parameter objects or device pointers.
        restored = make_model(upstream, dims, device)
        model.load_state_dict(restored.state_dict())
        del restored
        phase(receipt, "restored upstream timing input: validation")
        receipt["pre_timing"] = {EAGER_LABEL: {}, TUNED_LABEL: {}}
        for label, fn in ((EAGER_LABEL, model), (TUNED_LABEL, compiled)):
            validate_call(
                fn, timed_x, state, timed_ref, stream, receipt["pre_timing"][label], label
            )
        receipt["tuned"]["build_and_verify"] = "passed"
        phase(receipt, "paired CUDA-event timing")
        receipt["timing"] = time_pair(
            {EAGER_LABEL: model, TUNED_LABEL: compiled},
            timed_x,
            stream,
            args.samples,
            args.calls_per_sample,
        )
        receipt["timing"]["input_seed"] = UPSTREAM_SEEDS[0]


def configure_gpu(receipt: dict) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and a visible CUDA device are required")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    receipt["device"] = {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "torch_cuda_build": torch.version.cuda,
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    if (properties.major, properties.minor) != (
        12,
        0,
    ) or "rtx pro 6000 blackwell" not in properties.name.lower():
        raise RuntimeError("this GPU qualification requires RTX PRO 6000 Blackwell (sm_120)")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        raise RuntimeError("unset CUDA_LAUNCH_BLOCKING")
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        # PyTorch's newer precision API. Do not mix with legacy allow_tf32.
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        receipt["precision_flags"] = {
            "api": "fp32_precision",
            "global": torch.backends.fp32_precision,
            "matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn": torch.backends.cudnn.fp32_precision,
        }
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        receipt["precision_flags"] = {
            "api": "legacy",
            "matmul": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
    return device


def qualify(args: argparse.Namespace, receipt: dict) -> None:
    phase(receipt, "source pin")
    torch.set_default_dtype(torch.float32)
    if args.mode.startswith("cpu-"):
        torch.set_num_threads(args.cpu_threads)
        torch.set_float32_matmul_precision("highest")
    upstream = load_pinned(args.upstream)
    receipt["source_verified"] = True
    with torch.no_grad():
        if args.mode.startswith("cpu-"):
            receipt["device"] = {"name": "CPU", "threads": args.cpu_threads}
            run_cases(args, receipt, upstream, torch.device("cpu"))
            return
        phase(receipt, "GPU identity and precision policy")
        device = configure_gpu(receipt)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            if args.mode == "gpu-reference":
                run_cases(args, receipt, upstream, device, stream)
                return
            with tempfile.TemporaryDirectory(prefix="kb30-build-", dir=args.work_dir) as work:
                phase(receipt, "native author build/load")
                receipt["native_author"] = {
                    "arch": "sm_120",
                    "build": "not_started",
                    "load": "not_started",
                }
                native = AuthorControls(Path(work), args.nvcc, receipt["native_author"])
                try:
                    run_cases(args, receipt, upstream, device, stream, native)
                finally:
                    native.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cpu-smoke", "cpu-full", "gpu-reference", "gpu-controls", "gpu-full"),
        required=True,
    )
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--work-dir", type=Path, default=Path("/var/tmp"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--calls-per-sample", type=int, default=4)
    args = parser.parse_args()
    if min(args.samples, args.calls_per_sample, args.cpu_threads) < 1:
        parser.error("sample, call and thread counts must be positive")
    receipt: dict[str, Any] = {
        "schema": "kernelbench-l2-30-author-qualification/v1",
        "mode": args.mode,
        "status": "started",
        "dtype": "float32",
        "upstream_commit": SOURCE_COMMIT,
        "upstream_expected_sha256": SOURCE_SHA256,
        "source_verified": False,
        "upstream_path": str(args.upstream),
        "state_seed": STATE_SEED,
        "rtol": RTOL,
        "atol": ATOL,
        "model_candidate": "not_executed",
        "timing": "not_run",
        "pytorch": str(torch.__version__),
        "python": sys.version.split()[0],
        "harness_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("native_controls.py"),
                Path(__file__).with_name("controls.cu"),
            )
        },
    }
    with args.receipt.open("x") as file:
        try:
            qualify(args, receipt)
        except Exception as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            receipt.update(status="passed", phase="complete")
        finally:
            json.dump(receipt, file, indent=2, allow_nan=False)
            file.write("\n")
    print(f"kb30: {receipt['status']} ({args.mode}): {args.receipt}", file=sys.stderr, flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
