"""Closed ABI for direct paged GQA. No compiler, Driver API or fixture I/O."""

import ctypes as c
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import NamedTuple

ABI = "fi_gqa_h16_kv2_d128_p64_device_v1"
ENTRY = "fi_decode"
HEADS, KV_HEADS, WIDTH, PAGE = 16, 2, 128, 64
BLOCK = (128, 1, 1)
MAX_BYTES = 384 << 20
ALIGN = 256
PREFIX = bytes.fromhex("b77efa11de05493242392bf0eb391816") * 16
SUFFIX = bytes.fromhex("545c0991d4e98ab98b327761f3b70fc3") * 16
ZERO_BF16 = b"\x00\x00"
NEG_INF = struct.pack("<f", -math.inf)
F32_MAX = float.fromhex("0x1.fffffep+127")
POISON_BF16 = bytes.fromhex("c57f")
POISON_F32 = bytes.fromhex("a500c07f")
assert len(PREFIX) == len(SUFFIX) == ALIGN


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Descriptor:
    abi: str = ABI
    plan: str = "direct"
    geometry: tuple[str, ...] = ("head128",)

    def encode(self) -> bytes:
        return json.dumps(vars(self), sort_keys=True, separators=(",", ":")).encode()


DIRECT = Descriptor()


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate descriptor key: {key}")
        result[key] = value
    return result


def parse_descriptor(raw: bytes) -> Descriptor:
    if type(raw) is not bytes or not 0 < len(raw) <= 4096:
        raise ValueError("expected bounded UTF-8 descriptor bytes")
    obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
    if type(obj) is not dict or obj != json.loads(DIRECT.encode()):
        raise ValueError("only the exact direct/head128 schema is enabled (no split)")
    return DIRECT


@dataclass(frozen=True)
class Submission:
    source: bytes
    descriptor_bytes: bytes
    descriptor: Descriptor


def decode_submission(source: bytes, descriptor_bytes: bytes) -> Submission:
    """CPU-only framing/shape validation, not admission or permission to execute."""
    if type(source) is not bytes or not 0 < len(source) <= 128 << 10 or b"\x00" in source:
        raise ValueError("expected bounded NUL-free submission.cu bytes")
    text = source.decode("utf-8")
    if not text.strip() or any(line.lstrip().startswith("```") for line in text.splitlines()):
        raise ValueError("expected nonempty raw CUDA fragment, not fenced reply")
    return Submission(source, descriptor_bytes, parse_descriptor(descriptor_bytes))


class Arg(NamedTuple):
    name: str
    decl: str
    ctype: type


# Single ordered definition for emitted signature, launch packing and fixture binding.
ARGS = (
    Arg("Q", "const __nv_bfloat16*", c.c_uint64),
    Arg("K", "const __nv_bfloat16*", c.c_uint64),
    Arg("V", "const __nv_bfloat16*", c.c_uint64),
    Arg("I", "const int*", c.c_uint64),
    Arg("J", "const int*", c.c_uint64),
    Arg("T", "const int*", c.c_uint64),
    Arg("scale", "float", c.c_float),
    Arg("B", "int", c.c_int32),
    Arg("P", "int", c.c_int32),
    Arg("M", "int", c.c_int32),
    Arg("O", "__nv_bfloat16*", c.c_uint64),
    Arg("L", "float*", c.c_uint64),
)
POINTERS = tuple(arg.name for arg in ARGS if arg.ctype is c.c_uint64)
SIGNATURE = ", ".join(f"{arg.decl} {arg.name}" for arg in ARGS)
HEADER = f"""#include <cuda_bf16.h>
#include <math.h>
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(__nv_bfloat16) == 2);
__device__ __forceinline__ void {ENTRY}_body({SIGNATURE});
""".encode()
WRAPPER = (
    f'extern "C" __global__ void {ENTRY}({SIGNATURE}) {{\n'
    f"    {ENTRY}_body({', '.join(arg.name for arg in ARGS)});\n}}\n"
).encode()


def param_layout() -> tuple[tuple[int, int], ...]:
    offset, result = 0, []
    for arg in ARGS:
        align, size = c.alignment(arg.ctype), c.sizeof(arg.ctype)
        offset = (offset + align - 1) // align * align
        result.append((offset, size))
        offset += size
    return tuple(result)


PARAM_LAYOUT = param_layout()


@dataclass(frozen=True)
class Fixture:
    B: int
    P: int
    M: int
    scale: float
    Q: bytes
    K: bytes
    V: bytes
    I: bytes  # noqa: E741 (fixed ABI role name)
    J: bytes
    T: bytes


@dataclass(frozen=True)
class Launch:
    grid: tuple[int, int, int]
    block: tuple[int, int, int] = BLOCK
    shared_bytes: int = 0
    scratch_bytes: int = 0


@dataclass(frozen=True)
class Prepared:
    fixture: Fixture
    scale: float  # exactly the FP32 argument used by both worker and oracle
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    tail: tuple[int, ...]
    launch: Launch
    payloads: dict[str, bytes]
    fixture_sha256: str

    def arguments(self, addresses: dict[str, int]) -> tuple:
        if addresses.keys() != dict.fromkeys(POINTERS).keys():
            raise ValueError("expected exact ABI pointer roles")
        values = {**addresses, "scale": self.scale, **{k: getattr(self.fixture, k) for k in "BPM"}}
        return tuple(arg.ctype(values[arg.name]) for arg in ARGS)

    def token_counts(self) -> tuple[int, ...]:
        return tuple(
            (hi - lo - 1) * PAGE + tail if hi > lo else 0
            for lo, hi, tail in zip(self.indptr[:-1], self.indptr[1:], self.tail, strict=True)
        )


def ints(raw: bytes) -> tuple[int, ...]:
    return tuple(x for (x,) in struct.iter_unpack("<i", raw))


def bf16_max(raw: bytes) -> float:
    mag = max(x & 0x7FFF for (x,) in struct.iter_unpack("<H", raw))
    if mag >= 0x7F80:
        raise ValueError("BF16 inputs must be finite")
    return struct.unpack("<f", struct.pack("<I", mag << 16))[0]


def prepare(fixture: Fixture) -> Prepared:
    B, P, M = fixture.B, fixture.P, fixture.M
    for key, x, lower, upper in (("B", B, 1, 16), ("P", P, 1, 512), ("M", M, 0, 1024)):
        if type(x) is not int or not lower <= x <= upper:
            raise ValueError(f"{key} outside ABI domain")
    if type(fixture.scale) not in (float, int):
        raise ValueError("scale must be a finite positive FP32 value")
    try:
        scale = struct.unpack("<f", struct.pack("<f", fixture.scale))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("scale must fit FP32") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive FP32 value")

    # Python products cannot wrap; caps bound every signed device index and allocation.
    qsize, kvsize = 2 * B * HEADS * WIDTH, 2 * P * PAGE * KV_HEADS * WIDTH
    sizes = dict(Q=qsize, K=kvsize, V=kvsize, I=4 * (B + 1), J=4 * M, T=4 * B)
    inputs = {name: getattr(fixture, name) for name in sizes}
    for name, size in sizes.items():
        if type(inputs[name]) is not bytes or len(inputs[name]) != size:
            raise ValueError(f"{name}: need exactly {size} contiguous little-endian bytes")
    if sum(sizes.values()) + qsize + 4 * B * HEADS + 2 * ALIGN * len(POINTERS) + 4 > MAX_BYTES:
        raise ValueError("guarded device allocation cap exceeded")
    offsets, indices, tails = (ints(inputs[name]) for name in "IJT")
    rows = tuple(zip(offsets[:-1], offsets[1:], tails, strict=True))
    if offsets[0] != 0 or offsets[-1] != M or any(not 0 <= hi - lo <= 128 for lo, hi, _ in rows):
        raise ValueError("bad CSR indptr or too many pages in a row")
    if any(not 0 <= page < P for page in indices):
        raise ValueError("page ID outside [0,P)")
    for lo, hi, tail in rows:
        if (lo == hi and tail != 0) or (lo < hi and not 1 <= tail <= PAGE):
            raise ValueError("noncanonical row tail")
    q, k, v = (bf16_max(inputs[name]) for name in "QKV")
    dot = WIDTH * q * k
    if max(dot, dot * scale * math.log2(math.e), v * 128 * PAGE) > F32_MAX / 2:
        raise ValueError("unsafe FP32 accumulation range outside tested domain")

    h = hashlib.sha256(struct.pack("<iiif", B, P, M, scale))
    for raw in inputs.values():
        h.update(raw)
    payloads = {
        **inputs,
        "J": inputs["J"] if M else b"\xff" * 4,  # nonlogical dummy; not a valid page
        "O": POISON_BF16 * (B * HEADS * WIDTH),
        "L": POISON_F32 * (B * HEADS),
    }
    return Prepared(
        fixture, scale, offsets, indices, tails, Launch((HEADS, B, 1)), payloads, h.hexdigest()
    )
