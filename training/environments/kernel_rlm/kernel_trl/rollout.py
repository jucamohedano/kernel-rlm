"""TRL rollout adapter. Group policy and row conversion stay in collect."""

from pathlib import Path

from kernel_rollout.episode import EpisodeRequest, EpisodeResult, EpisodeRunner


class RolloutCollector:
    def __init__(self, client, adapter_name: str, tokenizer_path: str, **settings):
        self.runner = EpisodeRunner(client, adapter_name, tokenizer_path, **settings)

    async def run_episode(self, task: dict, directory: Path, policy_version: int) -> EpisodeResult:
        return await self.runner.run(EpisodeRequest(task, directory, policy_version))
