"""Seeded public input construction, without task definitions or output oracles."""

from __future__ import annotations


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
