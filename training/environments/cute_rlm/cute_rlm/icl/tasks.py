"""Fixed, public semantic cases and trusted PyTorch oracles for the ICL pilot.

Importing this module does not import Torch. Only the isolated evaluation worker
(or a CPU test fixture) supplies Torch to the input/oracle functions.
"""

from __future__ import annotations

from copy import deepcopy

CASE_SEED = 20261101


def _cases(shapes: list[dict], **scalars) -> list[dict]:
    return [dict(shape, **scalars, seed=CASE_SEED + i) for i, shape in enumerate(shapes)]


_VECTOR_CASES = [{"n": n} for n in (1, 127, 256, 1003)]
_MATRIX_CASES = [{"m": m, "n": n} for m, n in ((1, 1), (3, 7), (16, 32), (31, 129))]
_GEMM_CASES = [
    {"m": m, "k": k, "n": n} for m, k, n in ((1, 1, 1), (3, 5, 7), (16, 32, 16), (17, 33, 19))
]


def _task(task_id, family, description, args, inputs, output, cases, *, atol=1e-5, rtol=1e-5):
    return {
        "id": task_id,
        "family": family,
        "description": description,
        "interface": {
            "class_name": "ModelNew",
            "constructor_signature": "__init__(self)",
            "forward_signature": "forward(self, " + ", ".join(args) + ")",
            "forward_args": args,
            "inputs": inputs,
            "output": output,
        },
        "constructor_args": [],
        "cases": cases,
        "dtype": "float32",
        "atol": atol,
        "rtol": rtol,
        "api_packet_id": family + "-scalar-v1",
        "example_id": "generic-2d-add-v1",
        "admission": "cpu_fixture_checked_native_torch_and_gpu_packet_sufficiency_pending",
    }


_TASKS = [
    _task(
        "affine",
        "pointwise",
        "y = alpha*x + beta elementwise.",
        ["x", "alpha", "beta"],
        "Contiguous FP32 x of shape (N,); alpha and beta are Python floats.",
        "New contiguous FP32 tensor of shape (N,).",
        _cases(_VECTOR_CASES, alpha=-1.25, beta=0.375),
    ),
    _task(
        "binary_gate",
        "pointwise",
        "y = x * (1 / (1 + exp(-z))) elementwise.",
        ["x", "z"],
        "Contiguous FP32 x and z with the same shape (N,).",
        "New contiguous FP32 tensor of shape (N,).",
        _cases(_VECTOR_CASES),
    ),
    _task(
        "clamp",
        "pointwise",
        "y = min(max(x, lo), hi) elementwise; lo < hi.",
        ["x", "lo", "hi"],
        "Contiguous FP32 x of shape (N,); lo and hi are Python floats.",
        "New contiguous FP32 tensor of shape (N,).",
        _cases(_VECTOR_CASES, lo=-0.75, hi=0.5),
    ),
    _task(
        "row_bias",
        "indexing",
        "y[i,j] = x[i,j] + bias[j].",
        ["x", "bias"],
        "Contiguous FP32 x of shape (M,N), bias of shape (N,).",
        "New contiguous FP32 tensor of shape (M,N).",
        _cases(_MATRIX_CASES),
    ),
    _task(
        "transpose_copy",
        "indexing",
        "y[j,i] = x[i,j]. Return a copy, not a view.",
        ["x"],
        "Contiguous FP32 x of shape (M,N).",
        "New contiguous FP32 tensor of shape (N,M).",
        _cases(_MATRIX_CASES),
    ),
    _task(
        "strided_affine",
        "indexing",
        "y[i,j] = alpha*x[i,j] + beta; respect both input strides.",
        ["x", "alpha", "beta"],
        "FP32 x of shape (M,N), positive element strides (stride_m,stride_n). Storage offset zero. "
        "Input may be transposed or have gaps; use runtime tensor strides, not contiguous indexing. "
        "alpha and beta are Python floats.",
        "New contiguous FP32 tensor of shape (M,N).",
        _cases(
            [
                {"m": 1, "n": 1, "stride_m": 3, "stride_n": 2},
                {"m": 3, "n": 7, "stride_m": 1, "stride_n": 3},
                {"m": 16, "n": 32, "stride_m": 69, "stride_n": 2},
                {"m": 31, "n": 129, "stride_m": 391, "stride_n": 3},
            ],
            alpha=0.75,
            beta=-0.25,
        ),
    ),
    _task(
        "row_sum",
        "reduction",
        "y[i] = sum_j x[i,j].",
        ["x"],
        "Contiguous FP32 x of shape (M,N), N >= 1.",
        "New contiguous FP32 tensor of shape (M,).",
        _cases(_MATRIX_CASES),
        atol=2e-4,
        rtol=2e-4,
    ),
    _task(
        "row_max",
        "reduction",
        "y[i] = max_j x[i,j].",
        ["x"],
        "Contiguous FP32 x of shape (M,N), N >= 1; finite inputs.",
        "New contiguous FP32 tensor of shape (M,).",
        _cases(_MATRIX_CASES),
    ),
    _task(
        "row_normalize",
        "reduction",
        "y[i,j] = x[i,j] / sqrt((sum_k x[i,k]^2) / N + epsilon). "
        "The denominator is RMS with epsilon inside the square root; do not subtract the mean.",
        ["x", "epsilon"],
        "Contiguous FP32 x of shape (M,N), N >= 1; epsilon is a positive Python float.",
        "New contiguous FP32 tensor of shape (M,N).",
        _cases(_MATRIX_CASES, epsilon=1e-5),
        atol=2e-4,
        rtol=2e-4,
    ),
    _task(
        "matmul",
        "matrix",
        "C[i,j] = sum_k A[i,k] * B[k,j].",
        ["A", "B"],
        "Contiguous FP32 A of shape (M,K), B of shape (K,N). All dimensions positive.",
        "New contiguous FP32 tensor of shape (M,N).",
        _cases(_GEMM_CASES),
        atol=3e-4,
        rtol=3e-4,
    ),
    _task(
        "matvec",
        "matrix",
        "y[i] = sum_k A[i,k] * v[k].",
        ["A", "v"],
        "Contiguous FP32 A of shape (M,K), v of shape (K,). All dimensions positive.",
        "New contiguous FP32 tensor of shape (M,).",
        _cases([{"m": s["m"], "k": s["k"]} for s in _GEMM_CASES]),
        atol=3e-4,
        rtol=3e-4,
    ),
    _task(
        "matmul_bias",
        "matrix",
        "C[i,j] = sum_k A[i,k] * B[k,j] + bias[j].",
        ["A", "B", "bias"],
        "Contiguous FP32 A of shape (M,K), B of shape (K,N), bias of shape (N,).",
        "New contiguous FP32 tensor of shape (M,N).",
        _cases(_GEMM_CASES),
        atol=3e-4,
        rtol=3e-4,
    ),
]


def load_tasks() -> list[dict]:
    """Return independent copies so preparation cannot mutate the fixed manifest."""
    return deepcopy(_TASKS)


def get_task(task_id: str) -> dict:
    for task in _TASKS:
        if task["id"] == task_id:
            return deepcopy(task)
    raise KeyError(task_id)


def build_inputs(task: dict, case: dict, torch) -> list:
    """Generate CPU inputs with a per-case generator; preserve strides on transfer.

    Separate reference/candidate workers call this function with the same seed.
    The gapped strided view is not serialized as a contiguous tensor.
    """
    task_id = task["id"]
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32, device="cpu")

    if task_id in {"affine", "binary_gate", "clamp"}:
        x = randn(case["n"])
        if task_id == "binary_gate":
            return [x, randn(case["n"])]
        names = ("alpha", "beta") if task_id == "affine" else ("lo", "hi")
        return [x, *(case[name] for name in names)]
    if task_id in {"matmul", "matmul_bias", "matvec"}:
        A = randn(case["m"], case["k"])
        if task_id == "matvec":
            return [A, randn(case["k"])]
        args = [A, randn(case["k"], case["n"])]
        if task_id == "matmul_bias":
            args.append(randn(case["n"]))
        return args
    if task_id == "strided_affine":
        m, n, sm, sn = (case[key] for key in ("m", "n", "stride_m", "stride_n"))
        storage = randn(1 + (m - 1) * sm + (n - 1) * sn)
        x = storage.as_strided((m, n), (sm, sn))
        return [x, case["alpha"], case["beta"]]
    x = randn(case["m"], case["n"])
    if task_id == "row_bias":
        return [x, randn(case["n"])]
    if task_id == "row_normalize":
        return [x, case["epsilon"]]
    if task_id in {"transpose_copy", "row_sum", "row_max"}:
        return [x]
    raise KeyError(task_id)


# These expressions are trusted reference code, never included in a student prompt.
_ORACLES = {
    "affine": "x * alpha + beta",
    "binary_gate": "x * (1.0 / (1.0 + torch.exp(-z)))",
    "clamp": "torch.clamp(x, min=lo, max=hi)",
    "row_bias": "x + bias",
    "transpose_copy": "x.transpose(0, 1).contiguous().clone()",
    "strided_affine": "(x * alpha + beta).contiguous()",
    "row_sum": "torch.sum(x, dim=1)",
    "row_max": "torch.max(x, dim=1).values",
    "row_normalize": "x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + epsilon)",
    "matmul": "torch.matmul(A, B)",
    "matvec": "torch.mv(A, v)",
    "matmul_bias": "torch.matmul(A, B) + bias",
}


def reference_source(task: dict) -> str:
    """An auditable PyTorch Model; only the reference worker may execute it."""
    trusted = get_task(task["id"])
    signature = trusted["interface"]["forward_signature"]
    return (
        "import torch\n\nclass Model(torch.nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n"
        f"    def {signature}:\n        return {_ORACLES[trusted['id']]}\n"
    )


def expected_output(task: dict, inputs: list, torch):
    """Evaluate the oracle without importing or executing candidate code."""
    task_id = task["id"]
    if task_id == "affine":
        x, alpha, beta = inputs
        return x * alpha + beta
    if task_id == "strided_affine":
        x, alpha, beta = inputs
        return (x * alpha + beta).contiguous()
    if task_id == "binary_gate":
        x, z = inputs
        return x * (1.0 / (1.0 + torch.exp(-z)))
    if task_id == "clamp":
        x, lo, hi = inputs
        return torch.clamp(x, min=lo, max=hi)
    if task_id == "row_bias":
        x, bias = inputs
        return x + bias
    if task_id == "transpose_copy":
        return inputs[0].transpose(0, 1).contiguous().clone()
    if task_id == "row_sum":
        return torch.sum(inputs[0], dim=1)
    if task_id == "row_max":
        return torch.max(inputs[0], dim=1).values
    if task_id == "row_normalize":
        x, epsilon = inputs
        return x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + epsilon)
    if task_id == "matmul":
        return torch.matmul(*inputs)
    if task_id == "matvec":
        return torch.mv(*inputs)
    if task_id == "matmul_bias":
        A, B, bias = inputs
        return torch.matmul(A, B) + bias
    raise KeyError(task_id)
