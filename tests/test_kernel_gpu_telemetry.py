import json
from types import SimpleNamespace


def test_sample_gpu_stats_parses_nvidia_smi(monkeypatch):
    from training.environments.kernel_rlm.modal_app import gpu_telemetry

    monkeypatch.setattr(
        gpu_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, NVIDIA A100, 61, 42, 32000, 81920, 70, 255.5\n"
        ),
    )
    result = gpu_telemetry.sample_gpu_stats()
    assert result["gpus"] == [
        {
            "index": 0,
            "name": "NVIDIA A100",
            "gpu_utilization_percent": 61.0,
            "memory_utilization_percent": 42.0,
            "memory_used_mib": 32000.0,
            "memory_total_mib": 81920.0,
            "temperature_c": 70.0,
            "power_w": 255.5,
        }
    ]


def test_wait_with_gpu_sampling_persists_samples(monkeypatch, tmp_path):
    from training.environments.kernel_rlm.modal_app import gpu_telemetry

    class Process:
        returncode = None
        waits = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.waits += 1
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(
        gpu_telemetry,
        "sample_gpu_stats",
        lambda: {"timestamp_utc": "now", "gpus": [{"index": 0}]},
    )
    path = tmp_path / "telemetry" / "gpu_stats.jsonl"
    result = gpu_telemetry.wait_with_gpu_sampling(Process(), path, timeout_s=1)
    assert result == {"returncode": 0, "timed_out": False}
    assert json.loads(path.read_text()) == {
        "timestamp_utc": "now",
        "gpus": [{"index": 0}],
    }


def test_sample_gpu_stats_reports_missing_nvidia_smi(monkeypatch):
    from training.environments.kernel_rlm.modal_app import gpu_telemetry

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_telemetry.subprocess, "run", missing)
    result = gpu_telemetry.sample_gpu_stats()
    assert result["gpus"] == []
    assert "FileNotFoundError" in result["error"]
