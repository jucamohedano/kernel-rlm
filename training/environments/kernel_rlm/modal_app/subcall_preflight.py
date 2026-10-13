"""CPU-only validation of child loss exclusion through pinned Prime and V1."""

import json
from pathlib import Path

import modal

from .grpo_live_smoke import live_image
from .rollout_smoke import traces

app = modal.App("kernel-rlm-subcall-loss-preflight")


@app.function(
    image=live_image, volumes={"/traces": traces}, cpu=2, memory=8192, timeout=180, retries=0
)
def check() -> dict:
    import verifiers.v1 as vf
    from kernel_rlm.training_mask import mask_child_branches
    from prime_rl.orchestrator.algo.routing import assign_advantages
    from prime_rl.orchestrator.trajectories import trace_to_samples

    path = Path("/traces/frozen-probe/20261009T112652Z-298b409b/traces.jsonl")
    trace = vf.Trace.model_validate(json.loads(path.read_text().splitlines()[0])["traces"][0])
    before = sum(sum(sample.mask) for sample in trace_to_samples(trace))
    mask_child_branches(trace)
    trace = vf.Trace.model_validate_json(trace.model_dump_json())
    assign_advantages(trace, 1.0)
    samples = trace_to_samples(trace)
    after = sum(sum(sample.mask) for sample in samples)
    assert before == 2718 and after == 1816
    assert len(samples) == 1
    assert sum(sum(sample.advantages) for sample in samples) == 1816
    assert not any(trace.nodes[7].mask)
    assert trace.nodes[7].advantages is None
    assert trace.nodes[7].message.content
    report = {
        "status": "passed",
        "before_tokens": before,
        "root_loss_tokens": after,
        "excluded_child_tokens": before - after,
        "training_samples": len(samples),
        "note": "Synthetic unit advantage for mask testing, not a training update",
    }
    Path("/traces/subcall-loss-preflight.json").write_text(json.dumps(report, indent=2))
    traces.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), indent=2))
