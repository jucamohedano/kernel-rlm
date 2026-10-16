"""Verifiers V1 runtime adapter for this repository's Python REPL worker."""

import json

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.clients import ModelContext
from verifiers.v1.runtimes import ProgramResult, Runtime


class KernelHarnessConfig(vf.HarnessConfig):
    max_turns: int = Field(3, ge=1, le=8)
    max_tokens: int = Field(2048, ge=1, le=4096)
    max_context_tokens: int = Field(8192, ge=1024)
    model_timeout_s: float = Field(150, gt=0)
    enable_thinking: bool = False
    enable_subcall: bool = False
    max_subcalls: int = Field(1, ge=0, le=4)
    subcall_probe: bool = False


class KernelHarness(vf.Harness[KernelHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: ModelContext,
        trace: vf.Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError("This bounded REPL adapter does not support MCP tools")
        await runtime.write("task.json", data.model_dump_json().encode())
        return await runtime.run_program(
            ["python", "-m", "kernel_rlm_v1.program", "task.json"],
            {
                **self.config.resolved_env,
                "KERNEL_ENDPOINT": endpoint,
                "KERNEL_API_KEY": secret,
                "KERNEL_MODEL": ctx.model,
                "KERNEL_LIMITS": json.dumps(
                    {
                        "max_turns": self.config.max_turns,
                        "max_tokens": self.config.max_tokens,
                        "max_context_tokens": self.config.max_context_tokens,
                        "model_timeout_s": self.config.model_timeout_s,
                        "enable_thinking": self.config.enable_thinking,
                        "enable_subcall": self.config.enable_subcall,
                        "max_subcalls": self.config.max_subcalls,
                        "subcall_probe": self.config.subcall_probe,
                    }
                ),
            },
        )
