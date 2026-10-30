"""OOLONG scoring wrapper around native, managed recursive REPL sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from rlm_train.native import NativeCleanupError, NativeLimits, NativeRuntime
from rlm_train.proxy import SubLLMProxy
from rlm_train.repl.subprocess import SubprocessReplBackend


async def evaluate(
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    run_dir: Path,
    api_key: str | None = None,
    max_turns: int = 32,
    max_tokens: int = 8192,
    request_timeout: float = 600,
    episode_timeout: float = 1800,
    child_max_tokens: int = 4096,
    child_max_calls: int = 32,
    child_max_prompt_chars: int = 65536,
    child_max_concurrent: int = 2,
    max_depth: int = 2,
    max_live_children: int = 8,
    max_model_calls: int = 256,
    enable_thinking: bool = True,
    proxy_socket_path: Path | None = None,
    backend_factory=SubprocessReplBackend,
    client_factory=httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """Run fixed rows; only the host scorer receives answer metadata.

    ``child_max_calls`` bounds admitted recursive children across the episode tree.
    ``max_model_calls`` independently bounds all root, recursive and plain requests.
    A child at ``max_depth`` is a plain completion, without a Python REPL.
    """
    limits = NativeLimits(
        max_turns=max_turns,
        max_tokens=max_tokens,
        child_max_tokens=child_max_tokens,
        max_children=child_max_calls,
        max_live_children=max_live_children,
        max_concurrent_requests=child_max_concurrent,
        max_model_calls=max_model_calls,
        request_timeout=request_timeout,
        episode_timeout=episode_timeout,
        max_prompt_chars=child_max_prompt_chars,
        max_depth=max_depth,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    client = client_factory(
        base_url=base_url.rstrip("/") + "/",
        timeout=request_timeout,
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
    )
    proxy = SubLLMProxy(socket_path=proxy_socket_path)
    records = []
    try:
        await proxy.start()
        for index, row in enumerate(rows):
            runtime = NativeRuntime(
                client,
                model,
                proxy,
                limits=limits,
                backend_factory=backend_factory,
                enable_thinking=enable_thinking,
                event_path=run_dir / "native-events.jsonl",
            )
            try:
                record = await runtime.run(
                    row.get("root_prompt", row.get("question", "")), row.get("context", "")
                )
            except NativeCleanupError:
                (run_dir / f"task-{index:03d}.json").write_text(
                    json.dumps(runtime.records[runtime.root.id], ensure_ascii=False, indent=2)
                    + "\n"
                )
                raise
            record["example_id"] = row.get("id", row.get("example_id", index))
            if record.get("final_answer") is not None:
                from oolong.scoring import synth_score

                try:
                    record["score"] = synth_score(row, record["final_answer"])
                    record["correct"] = record["score"] == 1.0
                except Exception as error:
                    record.update(error_class="task", error=f"scoring failed: {error!r}")
            (run_dir / f"task-{index:03d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n"
            )
            records.append(record)
    finally:
        await proxy.stop()
        await client.aclose()
    summary = {
        "count": len(records),
        "denominator": len(records),
        "final_submissions": sum(r.get("final_answer") is not None for r in records),
        "correct": sum(r.get("correct", False) for r in records),
        "errors": {
            kind: sum(r.get("error_class") == kind for r in records)
            for kind in ("infrastructure", "protocol", "task", "budget", "cleanup")
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--episode-timeout", type=float, default=1800)
    parser.add_argument("--child-max-tokens", type=int, default=4096)
    parser.add_argument("--child-max-calls", type=int, default=32)
    parser.add_argument("--child-max-prompt-chars", type=int, default=65536)
    parser.add_argument("--child-max-concurrent", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-live-children", type=int, default=8)
    parser.add_argument("--max-model-calls", type=int, default=256)
    parser.add_argument("--proxy-socket", type=Path)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a JSON list")
    asyncio.run(
        evaluate(
            rows,
            base_url=args.base_url,
            model=args.model,
            run_dir=args.run_dir,
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            episode_timeout=args.episode_timeout,
            child_max_tokens=args.child_max_tokens,
            child_max_calls=args.child_max_calls,
            child_max_prompt_chars=args.child_max_prompt_chars,
            child_max_concurrent=args.child_max_concurrent,
            max_depth=args.max_depth,
            max_live_children=args.max_live_children,
            max_model_calls=args.max_model_calls,
            proxy_socket_path=args.proxy_socket,
            enable_thinking=not args.disable_thinking,
        )
    )


if __name__ == "__main__":
    main()
