"""Typed capacity outcomes, distinct from inference infrastructure failures."""


class ContextExhaustedError(ValueError):
    def __init__(self, prompt_tokens: int, context_tokens: int, requested_tokens: int):
        self.details = {
            "prompt_tokens": prompt_tokens,
            "context_tokens": context_tokens,
            "requested_tokens": requested_tokens,
        }
        super().__init__("Insufficient exact context headroom for a useful reply")


def context_exhaustion_details(error) -> dict | None:
    body = getattr(error, "body", None)
    if getattr(error, "status_code", None) != 400 or not isinstance(body, dict):
        return None
    payload = body.get("error", body)
    if not isinstance(payload, dict) or payload.get("code") != "context_exhausted":
        return None
    details = payload.get("details", {})
    if set(details) != {"prompt_tokens", "context_tokens", "requested_tokens"} or any(
        type(value) is not int or value < 0 for value in details.values()
    ):
        raise ValueError("Invalid context-exhaustion response")
    if details["context_tokens"] - details["prompt_tokens"] >= min(
        128, details["requested_tokens"]
    ):
        raise ValueError("Context-exhaustion response still has useful headroom")
    return details
