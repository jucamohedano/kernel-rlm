# Fixed Blackwell device-artifact gate

`vec_add_f32_v1` is a new author-only ABI. It is **not** the ragged host ABI or a
port of any candidate. The only supported descriptor pins `entry=vec_add`,
`target=sm_120`, `n=4`, `grid=[1,1,1]`, `block=[32,1,1]` and `shared_bytes=0`.
The types/order are fixed by the ABI: `[a:device_in_f32, b:device_in_f32,
out:device_out_f32, n:i32]`. No pointer, argument, source, artifact, flag, or
path override from a submission is supported. The hardware path only builds the
literal author source in `gate.py`, using pinned NVCC 13.3.73 and no user headers.
The ELF check is pinned to the observed CUDA executable header (`EI_OSABI=0x41`,
`EI_ABIVERSION=8`) and the base `sm_120` flag field; other layouts are unsupported.

Run `python -m pytest -q tests/test_blackwell_device_gate.py` from the repository
root in the existing project CPU environment. No CUDA library is opened by the
tests. They use synthetic ELF metadata (not a loadable module) and fake driver
calls. A mock receipt is never hardware evidence. A successful author smoke
would establish the data path only. Timing is always `not_run`; no candidate,
timing, scoring, external tool reply, or evidence-ingestion interfaces exist.

## Later, parent-only smoke (not executed by this patch)

Only after explicit approval for an exclusive test window and an agreed recovery
policy, verify the driver/toolkit/header closure, device UUID/driver build and
idle GPU in the preflight. Select the existing trusted absolute Python, NVCC,
and C++ compiler paths. Run from the repository root with a clean environment
and no inherited secrets or extra FDs, e.g.:

```sh
env -i PATH=/usr/bin:/bin /usr/bin/timeout --kill-after=5s 75s \
  /absolute/python -B -m training.environments.kernel_rlm.research.blackwell_device_gate.author_smoke \
  --approved-author-smoke --nvcc /absolute/nvcc --ccbin /absolute/g++ --device-uuid GPU-...
```

The runner checks ordinal 0's UUID and compute capability 12.0 before module
load. It creates a private context, launches once, synchronizes, and reads back
all three exact guarded allocations. Expect `[5,4,0,2]` with intact guards and
inputs and successful cleanup; retain the JSON receipt and preflight together.
The receipt's `driver_version` is the CUDA Driver API version, not the kernel
module build. Inspect the first receipt before manually doing a second fresh
run. Stop on any unexpected stage or missing receipt; on watchdog expiry,
quarantine, context/device/driver errors, or cleanup failure, withhold further
launches until the parent verifies recovery (including a known author smoke).
There is no automatic reset or retry. A killed process may leave no receipt and
does not imply the GPU recovered. Do not deliberately fault kernels. No VM,
driver, toolkit, package, or deployment changes are required or authorized here.

This is not a security boundary. The compiler and Driver API run trusted host
code and the GPU executes device code; guards and exact readback do not prove
safety, isolation, or arbitrary kernel correctness. Keep this author-only even
after successful smokes. Do not link generated host code or grant candidate
execution based on this patch.
