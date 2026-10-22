"""Loopback OpenAI relay into SkyRL's adapter-aware inference client."""

import asyncio
import copy
import json
import os
import secrets
import time
import uuid
from typing import Any

from aiohttp import web
from kernel_rlm.capacity import ContextExhaustedError
from kernel_rlm.child_tools import TOOL_CHILD_SYSTEM
from kernel_rlm.subcalls import CHILD_SYSTEM, SUBCALL_REPLY_TOKENS
from kernel_rlm.verifier_slot import INFERENCE_ADMISSION_TIMEOUT_S, inference_request_slot

from kernel_skyrl.contract import exact_sample


class CompletionBridge:
    def __init__(
        self,
        client: Any,
        adapter_name: str,
        context_limit: int,
        max_reply: int,
        policy_version: int,
        trace_path,
        timeout_s: float = 150,
        provenance: dict | None = None,
    ):
        if not adapter_name:
            raise ValueError("An explicit LoRA adapter name is required")
        self.client = client
        self.adapter_name = adapter_name
        self.context_limit = context_limit
        self.max_reply = max_reply
        self.timeout_s = timeout_s
        self.session_id = uuid.uuid4().hex
        self.api_key = secrets.token_urlsafe(32)
        self.policy_version = policy_version
        self.trace_path = trace_path
        self.provenance = dict(provenance or {})
        self.samples = []
        self.failures = []
        self.child_calls = 0
        self.child_progress = {}
        self.runner = None
        self.endpoint = None

    async def start(self) -> None:
        app = web.Application(client_max_size=2_000_000)
        app.router.add_post("/v1/chat/completions", self.complete)
        app.router.add_get("/v1/child-progress", self.progress_status)
        self.runner = web.AppRunner(app, shutdown_timeout=1, handler_cancellation=True)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        address = self.runner.addresses[0]
        self.endpoint = f"http://127.0.0.1:{address[1]}/v1"

    async def stop(self) -> None:
        try:
            if self.runner is not None:
                await self.runner.cleanup()
        finally:
            await asyncio.wait_for(self.client.finish_session(self.session_id), 5)

    async def progress_status(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.api_key}":
            raise web.HTTPUnauthorized()
        return web.json_response(
            {
                "observed_at_unix_seconds": time.time(),
                "children": list(self.child_progress.values()),
            }
        )

    def publish_child_progress(self, call_id, started, **fields):
        event = {
            **self.child_progress.get(call_id, {}),
            "call_id": call_id,
            "policy_version": self.policy_version,
            "elapsed_seconds": time.monotonic() - started,
            "updated_at_unix_seconds": time.time(),
            **fields,
        }
        self.child_progress[call_id] = event
        with self.trace_path.with_name("child-progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    async def complete(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.api_key}":
            raise web.HTTPUnauthorized()
        started = time.monotonic()
        record = {
            "producer": "kernel_inference_bridge",
            **self.provenance,
            "policy_version": self.policy_version,
        }
        try:
            body = copy.deepcopy(await request.json())
            headers = request.headers
            if headers.get("X-Kernel-Root-Turn-Id"):
                record["root_turn_id"] = headers["X-Kernel-Root-Turn-Id"]
            if headers.get("X-Kernel-Call-Id"):
                record["call_id"] = headers["X-Kernel-Call-Id"]
            if headers.get("X-Kernel-Assistant-Message-Index"):
                record["assistant_message_index"] = int(headers["X-Kernel-Assistant-Message-Index"])
            if headers.get("X-Kernel-Parent-Turn-Id"):
                record["parent_turn_id"] = headers["X-Kernel-Parent-Turn-Id"]
            if headers.get("X-Kernel-Parent-Tool-Call-Id"):
                record["parent_tool_call_id"] = headers["X-Kernel-Parent-Tool-Call-Id"]
            if headers.get("X-Kernel-Parent-Tool-Call-Id-Generated"):
                record["parent_tool_call_id_generated"] = (
                    headers["X-Kernel-Parent-Tool-Call-Id-Generated"] == "true"
                )
            if headers.get("X-Kernel-Parent-Action-Id"):
                record["parent_action_id"] = headers["X-Kernel-Parent-Action-Id"]
            messages = body["messages"]
            child = bool(
                messages
                and messages[0].get("role") == "system"
                and messages[0].get("content") in (CHILD_SYSTEM, TOOL_CHILD_SYSTEM)
            )
            limit = SUBCALL_REPLY_TOKENS if child else self.max_reply
            requested = body.pop("max_completion_tokens", body.pop("max_tokens", limit))
            if type(requested) is not int or requested <= 0 or body.get("stream"):
                raise ValueError("Expected a bounded non-streaming completion")
            body.update(
                model=self.adapter_name,
                session_id=self.session_id,
                n=1,
                max_tokens=min(requested, limit),
                logprobs=True,
                return_token_ids=True,
                return_tokens_as_token_ids=True,
                temperature=1.0,
                top_p=1.0,
                cache_salt=f"kernel-{self.policy_version}",
            )
            record.update(branch="child" if child else "root", request=copy.deepcopy(body))
            if not record.get("call_id"):
                record["call_id"] = uuid.uuid4().hex
            record["model_call_id"] = record["call_id"]
            if child:
                self.publish_child_progress(
                    record["call_id"], started, stage="queued", completion_tokens=0
                )
            async with inference_request_slot(
                INFERENCE_ADMISSION_TIMEOUT_S,
                enabled=bool(os.environ.get("KERNEL_INFERENCE_ADMIN_URL")),
            ) as admission_wait:
                record["admission_wait_seconds"] = admission_wait
                streaming = getattr(self.client, "chat_completion_with_progress", None)
                if child:
                    self.publish_child_progress(
                        record["call_id"],
                        started,
                        stage="admitted",
                        admission_wait_seconds=admission_wait,
                        token_progress_available=streaming is not None,
                    )
                if child and streaming is not None:

                    def progress(event):
                        self.publish_child_progress(record["call_id"], started, **event)

                    completion = streaming({"json": body}, progress)
                else:
                    completion = self.client.chat_completion({"json": body})
                response = await asyncio.wait_for(completion, self.timeout_s)
            record["response"] = response
            sample = exact_sample(
                response,
                self.context_limit,
                {
                    key: record[key]
                    for key in (
                        "execution_run_id",
                        "group_id",
                        "attempt_id",
                        "episode_id",
                        "policy_version",
                        "root_turn_id",
                        "call_id",
                        "model_call_id",
                    )
                    if key in record
                },
            )
            if len(sample.response_ids) > body["max_tokens"]:
                raise ValueError("Inference exceeded this call's output cap")
            if child:
                self.child_calls += 1
                self.publish_child_progress(
                    record["call_id"],
                    started,
                    stage="completed",
                    completion_tokens=len(sample.response_ids),
                    finish_reason=sample.stop_reason,
                )
            else:
                self.samples.append(sample)
            return web.json_response(response)
        except asyncio.CancelledError:
            record.update(outcome="cancelled", error="Caller disconnected or bridge stopped")
            if record.get("branch") != "child":
                self.failures.append("Root completion cancelled before delivery")
            raise
        except TimeoutError as error:
            record.update(outcome="timeout", error=f"{type(error).__name__}: {error}")
            self.failures.append(record["error"])
            return web.json_response({"error": {"message": record["error"]}}, status=502)
        except ContextExhaustedError as error:
            record.update(outcome="context_exhausted", context_exhaustion=error.details)
            return web.json_response(
                {
                    "error": {
                        "code": "context_exhausted",
                        "message": str(error),
                        "details": error.details,
                    }
                },
                status=400,
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.failures.append(message)
            record["error"] = message
            return web.json_response({"error": {"message": message}}, status=502)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            if "call_id" in record and (record.get("error") or "outcome" in record):
                self.publish_child_progress(
                    record["call_id"],
                    started,
                    stage=record.get("outcome", "failed"),
                    error=record.get("error"),
                )
            with self.trace_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
