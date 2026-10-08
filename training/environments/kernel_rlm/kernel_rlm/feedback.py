"""Generate bounded CUDA-feedback helpers for the persistent REPL worker."""

from __future__ import annotations

import json
import textwrap


def build_local_feedback_bootstrap(
    reference_code: str,
    *,
    max_calls: int = 2,
    timeout_s: float = 120.0,
) -> str:
    """Build worker code that runs the verifier in the worker's CUDA environment."""
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    return textwrap.dedent(
        f"""
        import json as _kernel_feedback_json
        from kernel_rlm.verifier import verify_cuda as _kernel_feedback_verify

        _kernel_feedback_reference = {json.dumps(reference_code)}
        _kernel_feedback_calls = 0

        def cuda_feedback(source):
            global _kernel_feedback_calls
            if _kernel_feedback_calls >= {max_calls}:
                result = {{"error": "CUDA feedback call budget exhausted"}}
            else:
                _kernel_feedback_calls += 1
                try:
                    result = _kernel_feedback_verify(
                        _kernel_feedback_reference,
                        str(source),
                        n_trials=1,
                        timeout_s={float(timeout_s)!r},
                        benchmark=False,
                    )
                except Exception as _kernel_feedback_error:
                    result = {{
                        "compiled": False,
                        "extension_used": False,
                        "correct": False,
                        "max_diff": None,
                        "error": (
                            type(_kernel_feedback_error).__name__
                            + ": "
                            + str(_kernel_feedback_error)
                        ),
                    }}
            result = {{
                "compiled": False,
                "extension_used": False,
                "correct": False,
                "max_diff": None,
                "speedup": None,
                "ref_ms": None,
                "kernel_ms": None,
                "error": "",
                **result,
                "feedback_calls_used": _kernel_feedback_calls,
                "feedback_calls_remaining": {max_calls} - _kernel_feedback_calls,
            }}
            print("CUDA feedback: " + _kernel_feedback_json.dumps(result, sort_keys=True))
            return result
        """
    ).strip()


def build_http_feedback_bootstrap(
    feedback_url: str,
    *,
    max_calls: int = 2,
    timeout_s: float = 180.0,
) -> str:
    """Build worker code that asks a separate GPU verifier over localhost HTTP."""
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    return textwrap.dedent(
        f"""
        import json as _kernel_feedback_json
        from urllib.request import Request as _kernel_feedback_request
        from urllib.request import urlopen as _kernel_feedback_urlopen

        _kernel_feedback_url = {feedback_url!r}
        _kernel_feedback_calls = 0

        def cuda_feedback(source):
            global _kernel_feedback_calls
            if _kernel_feedback_calls >= {max_calls}:
                result = {{"error": "CUDA feedback call budget exhausted"}}
            else:
                _kernel_feedback_calls += 1
                try:
                    payload = _kernel_feedback_json.dumps(
                        {{"source": str(source)}}
                    ).encode()
                    request = _kernel_feedback_request(
                        _kernel_feedback_url,
                        data=payload,
                        headers={{"Content-Type": "application/json"}},
                        method="POST",
                    )
                    with _kernel_feedback_urlopen(
                        request, timeout={float(timeout_s)!r}
                    ) as response:
                        result = _kernel_feedback_json.loads(response.read())
                except Exception as _kernel_feedback_error:
                    result = {{
                        "compiled": False,
                        "extension_used": False,
                        "correct": False,
                        "max_diff": None,
                        "error": (
                            type(_kernel_feedback_error).__name__
                            + ": "
                            + str(_kernel_feedback_error)
                        ),
                    }}
            result = {{
                "compiled": False,
                "extension_used": False,
                "correct": False,
                "max_diff": None,
                "speedup": None,
                "ref_ms": None,
                "kernel_ms": None,
                "error": "",
                **result,
                "feedback_calls_used": _kernel_feedback_calls,
                "feedback_calls_remaining": {max_calls} - _kernel_feedback_calls,
            }}
            print("CUDA feedback: " + _kernel_feedback_json.dumps(result, sort_keys=True))
            return result
        """
    ).strip()
