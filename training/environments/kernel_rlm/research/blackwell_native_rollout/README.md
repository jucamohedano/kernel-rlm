# Native Blackwell rollout (generation only)

This adapter runs one attempt through the pinned Harness-CG `RLMTrainEnv`
with the unchanged ragged pilot's isolated REPL worker and depth-1 child
broker. Only the controller's public task text is loaded into `context`.
No grader, fixtures, solution, compiler or device is exposed to the REPL.
The task text must give the model the public device-body and descriptor
contract. The launcher provides no strategy, fallback answer, or feedback.

## Controller interface

From `training.environments.kernel_rlm.research.blackwell_native_rollout.launcher`:

- `Budget(root_turns, child_calls, output_tokens, wall_seconds)` requires
  explicit caps within 2–8 root turns, 0–4 child prompts (0 disables the
  broker), 1–2048 output tokens **per root or child response**, and 1–900
  wall seconds. The controller chooses the same budget for matched attempts.
- `await generate_one(task_id=..., public_context=..., budget=..., out_dir=...,
  harness_checkout=..., python_runtime=..., client=..., model=...)` takes a
  trusted client; use a scripted one for CPU checks. Paths are `Path` objects.
- Optional `await run(task_id=..., public_context=..., budget=..., out_dir=...,
  harness_checkout=..., python_runtime=..., serving_manifest=..., api_key=...)`
  takes the existing pinned ragged-model serving manifest (including literal
  `http://127.0.0.1:<port>/v1`) and a controller-only key. Set
  `RLM_TRAIN_SUBCALL_TIMEOUT_S=80` if children are enabled. Verify the actual
  service independently; manifest validation is not model identity proof.
  This does not start, activate, or qualify a service. No automatic retries.

Use the unchanged Harness-CG checkout at
`5272afdcb25a156c1b9b89cb479ae0604af249d7`, its controller Python environment,
standalone CPython 3.13, and supported rootless bubblewrap. The output path
must be fresh, private, and outside this worktree.

## Native handoff

Commit `answer["content"]` as a JSON **text string** with exactly two string
fields: `source` (raw CUDA device-body text, ≤128 KiB) and `descriptor`
(separate JSON object text, ≤4 KiB). Set `answer["ready"] = True` to commit.
`parse_submission(text)` returns exact UTF-8 `answer`, `source`, and
`descriptor_text` bytes plus a plain JSON `descriptor` dictionary. It does
not validate the task-specific ABI, plans, presets, or source semantics.

A successful preliminary parse retains `answer.json`, `submission.cu`, and
`descriptor.json`, with SHA-256 digests in `result.json`. The result is still
`submitted_held_for_review`, with `kernel_verdict: null`. Empty/missing
commits, malformed commits (including fenced text), and transport failure
or timeout are distinct outcomes; none produce submission files. The receipt
records visible root messages and bounded native child logs/broker counts
when the native state is available. These logs are not exact wire captures,
and visible REPL stdout is model-controlled. `task-context.txt` and
`intent.json` bind the public brief, task ID, pins, and caps. No device score
or permission to compile follows from these files.

## CPU checks

Run `python -m pytest -q -o pythonpath= tests/test_blackwell_native_rollout.py`
from the repo root in the pinned Harness-CG environment. Set
`HARNESS_CG_CHECKOUT` and `RAGGED_PYTHON_RUNTIME` for the integration cases;
they use the real pinned turn loop and sandbox with in-process mock model
responses, never a live inference endpoint or GPU.
