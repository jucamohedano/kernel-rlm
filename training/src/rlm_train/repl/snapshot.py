"""Data-only snapshots of supported REPL values; never pickle or replay actions."""

from __future__ import annotations

import ast
import base64
import importlib
import math
import types

SAFE_MODULES = frozenset(
    {
        "ast",
        "json",
        "math",
        "re",
        "collections",
        "itertools",
        "functools",
        "statistics",
        "textwrap",
        "typing",
        "hashlib",
        "operator",
        "copy",
    }
)


def encode_namespace(variables: dict) -> dict:
    nodes = []
    seen = {}
    unsupported = []

    def encode(value):
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise TypeError("nonfinite float")
            return value
        if isinstance(value, types.ModuleType):
            if value.__name__.split(".")[0] not in SAFE_MODULES:
                raise TypeError("module outside recovery allowlist")
            return {"module": value.__name__}
        if type(value) not in (dict, list, tuple, set, frozenset, bytes) and not isinstance(
            value, ast.AST
        ):
            raise TypeError(type(value).__name__)
        identity = id(value)
        if identity in seen:
            return {"ref": seen[identity]}
        index = len(nodes)
        seen[identity] = index
        node = {}
        nodes.append(node)
        if isinstance(value, ast.AST):
            node.update(
                kind="ast",
                name=type(value).__name__,
                fields={name: encode(child) for name, child in ast.iter_fields(value)},
                attributes={
                    name: encode(getattr(value, name))
                    for name in value._attributes
                    if hasattr(value, name)
                },
            )
        elif type(value) is dict:
            node.update(kind="dict", items=[[encode(k), encode(v)] for k, v in value.items()])
        elif type(value) is bytes:
            node.update(kind="bytes", value=base64.b64encode(value).decode())
        else:
            node.update(kind=type(value).__name__, items=[encode(child) for child in value])
        return {"ref": index}

    encoded = {}
    for name, value in variables.items():
        try:
            encoded[name] = encode(value)
        except (TypeError, ValueError, RecursionError) as error:
            unsupported.append({"name": name, "type": type(value).__name__, "reason": str(error)})
    snapshot = {
        "schema_version": 1,
        "variables": encoded,
        "nodes": nodes,
        "imports": [],
        "unsupported": unsupported,
    }
    if not unsupported:
        try:
            decode_namespace(snapshot)
        except (TypeError, ValueError, RecursionError) as error:
            unsupported.append({"name": "<namespace>", "type": "graph", "reason": str(error)})
    return snapshot


def decode_namespace(snapshot: dict) -> dict:
    if snapshot.get("schema_version") != 1 or snapshot.get("unsupported"):
        raise ValueError("REPL checkpoint contains unsupported state")
    nodes = snapshot["nodes"]
    decoded = {}
    active = set()

    def decode(value):
        if not isinstance(value, dict):
            return value
        if "module" in value:
            name = value["module"]
            if name.split(".")[0] not in SAFE_MODULES:
                raise ValueError("checkpoint import outside recovery allowlist")
            return importlib.import_module(name)
        index = value["ref"]
        if index in decoded:
            return decoded[index]
        if index in active:
            raise ValueError("cyclic immutable checkpoint value")
        active.add(index)
        node = nodes[index]
        kind = node["kind"]
        if kind == "dict":
            result = {}
            decoded[index] = result
            result.update((decode(k), decode(v)) for k, v in node["items"])
        elif kind == "list":
            result = []
            decoded[index] = result
            result.extend(decode(child) for child in node["items"])
        elif kind == "ast":
            cls = getattr(ast, node["name"])
            if not isinstance(cls, type) or not issubclass(cls, ast.AST):
                raise ValueError("invalid checkpoint AST type")
            result = cls()
            decoded[index] = result
            for name, child in {**node["fields"], **node["attributes"]}.items():
                setattr(result, name, decode(child))
        elif kind == "bytes":
            result = base64.b64decode(node["value"], validate=True)
        elif kind in {"tuple", "set", "frozenset"}:
            constructor = {"tuple": tuple, "set": set, "frozenset": frozenset}[kind]
            result = constructor(decode(child) for child in node["items"])
        else:
            raise ValueError("unsupported checkpoint value kind")
        decoded[index] = result
        active.remove(index)
        return result

    return {name: decode(value) for name, value in snapshot["variables"].items()}
