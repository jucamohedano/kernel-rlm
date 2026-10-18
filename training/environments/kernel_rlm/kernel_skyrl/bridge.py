"""Loopback OpenAI relay into SkyRL's adapter-aware inference client."""

import asyncio
import copy
import json
import secrets
import time
import uuid
from typing import Any

from aiohttp import web
from kernel_rlm.subcalls import CHILD_SYSTEM

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
        self.samples = []
        self.failures = []
        self.child_calls = 0
        self.runner = None
        self.endpoint = None

    async def start(self) -> None:
        app = web.Application(client_max_size=2_000_000)
        app.router.add_post("/v1/chat/completions", self.complete)
        self.runner = web.AppRunner(app, shutdown_timeout=1)
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

    async def complete(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.api_key}":
            raise web.HTTPUnauthorized()
        started = time.monotonic()
        record = {"policy_version": self.policy_version}
        try:
            body = copy.deepcopy(await request.json())
            messages = body["messages"]
            child = bool(
                messages
                and messages[0].get("role") == "system"
                and messages[0].get("content") == CHILD_SYSTEM
            )
            limit = min(self.max_reply, 2048) if child else self.max_reply
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
            response = await asyncio.wait_for(
                self.client.chat_completion({"json": body}), self.timeout_s
            )
            record["response"] = response
            sample = exact_sample(response, self.context_limit)
            if len(sample.response_ids) > body["max_tokens"]:
                raise ValueError("Inference exceeded this call's output cap")
            if child:
                self.child_calls += 1
            else:
                self.samples.append(sample)
            return web.json_response(response)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.failures.append(message)
            record["error"] = message
            return web.json_response({"error": {"message": message}}, status=502)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            with self.trace_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
