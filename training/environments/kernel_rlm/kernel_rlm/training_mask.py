"""Exclude child branches from training while retaining their trace evidence."""

from kernel_rlm.subcalls import CHILD_SYSTEM


def mask_child_branches(trace) -> None:
    nodes = getattr(trace, "nodes", [])
    roots = []
    excluded = trace.info.setdefault("child_loss_exclusions", {})
    for index, node in enumerate(nodes):
        if node.parent is None:
            root = index
            if index and node.message.content != CHILD_SYSTEM:
                raise ValueError("Unknown independent trace branch; cannot assign root-only loss")
        else:
            if not 0 <= node.parent < index:
                raise ValueError("Invalid trace parent ordering")
            root = roots[node.parent]
        roots.append(root)
        if root == 0:
            continue
        if any(node.mask):
            excluded[str(index)] = {"mask": list(node.mask), "logprobs": list(node.logprobs)}
        node.mask = [False] * len(node.token_ids)
        node.logprobs = []
        node.advantages = None
        node.reference_logprobs = None
        node.loss_weights = None
        node.kept_tokens = None
