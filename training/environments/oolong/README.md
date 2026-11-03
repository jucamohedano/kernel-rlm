# OOLONG native recursive evaluation

The generic training environment remains available through `oolong.load_environment`.
The native evaluator uses `rlm_train.native.NativeRuntime` and keeps scoring on the
host. It sends only the task question to the model; the task context is loaded in
the root Python worker and gold metadata is never given to a model or code worker.

## Recursive REPLs

Every recursive session below the depth cutoff owns a persistent subprocess REPL,
its supplied context, Python variables and conversation/action history. Parent and
sibling variables are not copied. Child state lasts through that child's turns;
there is no follow-up or resume API after it finishes.

The only model-facing external tool is native `repl(code)`, at every recursive
REPL depth. Inside Python:

```python
plain = llm_query("Classify this snippet")
child = rlm_query("Solve this supplied subtask", blocking=False)
parent_work = len(context.splitlines())
state = rlm_status(child)
child_answer = rlm_wait(child, timeout_s=30)
answer["content"] = child_answer
answer["ready"] = True
```

- `llm_query` and `llm_query_batched` make plain completions with no tools.
- `rlm_query` blocks and returns an answer by default. `blocking=False` returns an
  episode-scoped handle. `rlm_query_batched` returns ordered answers.
- `rlm_status`, `rlm_result` and `rlm_wait` inspect or collect a caller's children.
  A wait timeout does not cancel the child. Unfinished/failed results raise errors.
- Root depth is zero. With `max_depth=2`, children at depth one have REPLs; calls at
  depth two are plain completions. No Python REPL is created at the cutoff.
- Each worker's `history` is a snapshot of its own prior model/tool messages. This
  is not a shared parent transcript or a channel for modifying another session.

## Limits and cleanup

The host owns the entire child tree and derives depth and ownership from registered
sessions, not worker-provided depth values. Admission is cumulative across the
whole episode. Completed/failed children still count until the episode ends.

Live child-session capacity and concurrent model requests are separate limits.
Full live-session capacity rejects a spawn; waiting ancestors still consume live
slots. No model-request permit is held while executing Python or waiting for a
child. Global model-call and per-session turn/output limits also apply. The whole
episode deadline includes nested work.

Parent finalization, errors, timeout and runtime close stop admission and clean up
descendants. Shutdown escalates from a cooperative request to terminate/kill and
requires confirmed worker exit. Cleanup failure is reported; it is not a successful
episode and must not be reused. Closing a local HTTP request or worker does **not**
prove the inference server stopped its generation.

These are separate Python processes, not a complete security sandbox. A privileged
launcher must still restrict generated code's filesystem, network and GPU access.
Model HTTP clients run on the host; the broker does not send credentials to code
workers. The subprocess backend inherits its launch environment, so a privileged
launcher must remove API keys and other secrets before starting workers and enable
`RLM_TRAIN_READONLY_CONTEXT=1` to reject context rebinding. The prompt's read-only
instruction alone is not a security boundary. No OpenEnv or smolagents runtime
dependency is introduced.

## Run a new evaluation

Use a new immutable run ID and the experiment factory before any real model run.
The runtime does not authorize a GPU experiment or capture complete source/runtime
provenance by itself.

```bash
PYTHONPATH=training/src:training/environments/oolong \
  python -m oolong.native_eval tasks.json \
  --base-url http://127.0.0.1:8000/v1 --model MODEL \
  --run-dir /path/to/new-run --max-depth 2
```

New runtime defaults are 32 turns/session, 8,192 root output tokens, 4,096 child
output tokens, 32 admitted children, eight live children, two concurrent requests,
256 total model requests, a 600-second request timeout, a 1,800-second episode
deadline and 65,536 characters per child prompt. These are **not** retroactive
changes to the stopped OOLONG diagnostic run.

Each task JSON includes the root and child session records. `native-events.jsonl`
records admission, request start before awaiting a response, raw responses,
executed actions, terminal outcomes and cleanup. Session/parent/depth/spawn IDs
are immutable. Count response IDs once for usage; tree/trajectory copies are not
additional inference. No logprobs or full logits are requested by this evaluator.

Design reference: [OpenEnv PR #1527](https://github.com/openenv/OpenEnv/pull/1527),
reviewed at `cb5812fc67c28014fb6a7034839f6c7d8050eac9`. This implementation keeps
native tool calls instead of using that PR's fenced-code runner. Selective
model-facing cancellation and follow-up messages are out of scope.
