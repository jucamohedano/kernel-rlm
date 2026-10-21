"""Audit saved thinking tokens and Prime loss samples without a GPU."""

import json
from collections import Counter
from pathlib import Path

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .grpo_live_smoke import live_image
from .rollout_smoke import traces

app = modal.App("kernel-rlm-thinking-audit")


@app.function(
    image=live_image,
    volumes={"/models": models, "/traces": traces},
    cpu=2,
    memory=8192,
    timeout=180,
    retries=0,
)
def audit(run_id: str):
    import verifiers.v1 as vf
    from huggingface_hub import snapshot_download
    from kernel_rlm.training_mask import mask_child_branches
    from prime_rl.orchestrator.algo.routing import assign_advantages
    from prime_rl.orchestrator.trajectories import trace_to_samples
    from transformers import AutoTokenizer

    directory = Path("/traces/shared-gpu-probe") / run_id
    data = json.loads((directory / "traces.jsonl").read_text().splitlines()[0])["traces"][0]
    trace = vf.Trace.model_validate(data)
    mask_child_branches(trace)
    assign_advantages(trace, 1.0)
    samples = trace_to_samples(trace)
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    calls = []
    for index, node in enumerate(trace.nodes):
        tokens = [
            token for token, sampled in zip(node.token_ids, node.mask, strict=True) if sampled
        ]
        if not tokens:
            continue
        decoded = tokenizer.decode(tokens)
        calls.append(
            {
                "node": index,
                "sampled_tokens": len(tokens),
                "reasoning_present": bool(data["nodes"][index]["message"].get("reasoning_content")),
                "decoded_start": decoded[:500],
                "decoded_end": decoded[-500:],
                "most_common_tokens": [
                    (tokenizer.decode([token]), count)
                    for token, count in Counter(tokens).most_common(3)
                ],
            }
        )
    report = {
        "run_id": run_id,
        "samples": len(samples),
        "loss_tokens": sum(sum(sample.mask) for sample in samples),
        "sampled_tokens": sum(call["sampled_tokens"] for call in calls),
        "calls": calls,
    }
    assert report["loss_tokens"] == report["sampled_tokens"]
    (directory / "thinking-audit.json").write_text(json.dumps(report, indent=2))
    traces.commit()
    return report


@app.local_entrypoint()
def main(run_id: str = "20261013T101758Z"):
    print(json.dumps(audit.remote(run_id), indent=2))
