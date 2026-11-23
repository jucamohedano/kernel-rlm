"""Fixed synthetic fixtures and independent, two-pass CPU oracle (no GPU code)."""

import math
import random
import struct
from dataclasses import replace

from .device_abi import HEADS, KV_HEADS, NEG_INF, PAGE, WIDTH, ZERO_BF16, Fixture, Prepared

CASES = ("analytic", "irregular", "mixed-empty", "all-empty")
ORACLE = "fp64-two-pass-natural-exp-bf16-rne/1"


def bf16(value: float) -> int:
    # Round to nearest even, directly to BF16 (including BF16 subnormals).
    # Canonical synthetic ranges fit BF16. Keep the sign of negative zero.
    sign = 0x8000 if math.copysign(1.0, value) < 0 else 0
    value = abs(value)
    if not value:
        return sign
    _, e = math.frexp(value)
    e = max(-125, e)
    mantissa = round(math.ldexp(value, 8 - e))
    return sign | (((e + 125) << 7) + mantissa)


def unpack_bf16(raw: bytes) -> tuple[float, ...]:
    # BF16 in little-endian floats is the high halfword, with the low halfword zero.
    wide = bytearray(len(raw) * 2)
    wide[2::4] = raw[::2]
    wide[3::4] = raw[1::2]
    return struct.unpack(f"<{len(raw) // 2}f", wide)


def pack_i32(values: tuple[int, ...]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def analytic() -> Fixture:
    # Each Q has only one nonzero component. Logits for the first two tokens
    # of page 1 are [0.5, 1.0] for head group 0 and [-0.5, 0.0] for group 1.
    # The two rows share this page but consume different numbers of tokens.
    B, P = 2, 2
    q = struct.pack("<H", bf16(2)) + ZERO_BF16 * (WIDTH - 1)
    k = bytearray(struct.pack("<H", bf16(16)) * (P * PAGE * KV_HEADS * WIDTH))
    v = bytearray(struct.pack("<H", bf16(-16)) * (P * PAGE * KV_HEADS * WIDTH))
    for t in range(2):
        for kv in range(KV_HEADS):
            offset = ((PAGE + t) * KV_HEADS + kv) * WIDTH
            struct.pack_into("<H", k, 2 * offset, bf16(1 + t - 2 * kv))
            for d in range(WIDTH):
                val = (2 + (d % 8) / 8) * (-1 if t else 1) - kv
                struct.pack_into("<H", v, 2 * (offset + d), bf16(val))
    return Fixture(
        B,
        P,
        2,
        0.25,
        q * (B * HEADS),
        bytes(k),
        bytes(v),
        pack_i32((0, 1, 2)),
        pack_i32((1, 1)),
        pack_i32((1, 2)),
    )


def irregular() -> Fixture:
    B, P = 5, 8
    rng = random.Random(411)
    values = tuple(struct.pack("<H", bf16((i - 8) / 8)) for i in range(16))

    def field(n: int) -> bytes:
        return b"".join(values[rng.getrandbits(4)] for _ in range(n))

    q = field(B * HEADS * WIDTH)
    k, v = (bytearray(field(P * PAGE * KV_HEADS * WIDTH)) for _ in range(2))
    for page, used in ((5, 1), (1, 63), (3, 1), (2, 25), (0, 0)):
        lo = (page * PAGE + used) * KV_HEADS * WIDTH
        hi = (page + 1) * PAGE * KV_HEADS * WIDTH
        k[2 * lo : 2 * hi] = struct.pack("<H", bf16(16)) * (hi - lo)
        v[2 * lo : 2 * hi] = struct.pack("<H", bf16(-16)) * (hi - lo)
    return Fixture(
        B,
        P,
        8,
        0.125,
        q,
        bytes(k),
        bytes(v),
        pack_i32((0, 1, 2, 3, 5, 8)),
        pack_i32((5, 1, 6, 6, 3, 7, 7, 2)),
        pack_i32((1, 63, 64, 1, 25)),
    )


def fixture(name: str) -> Fixture:
    if name == "analytic":
        return analytic()
    if name == "all-empty":
        case = analytic()
        return replace(case, M=0, I=pack_i32((0, 0, 0)), J=b"", T=pack_i32((0, 0)))
    if name == "irregular":
        return irregular()
    if name == "mixed-empty":
        case = irregular()
        return replace(
            case, M=7, I=pack_i32((0, 0, 1, 2, 4, 7)), J=case.J[4:], T=pack_i32((0, 63, 64, 1, 25))
        )
    raise ValueError("unknown fixed author case")


def oracle(prepared: Prepared) -> tuple[bytes, bytes]:
    """Gather by logical token number; use FP64 and a stable two-pass softmax."""
    f = prepared.fixture
    q, k, v = (unpack_bf16(getattr(f, name)) for name in "QKV")
    out_bytes, lse_bytes = bytearray(), bytearray()
    for b, n in enumerate(prepared.token_counts()):
        if not n:
            out_bytes.extend(ZERO_BF16 * (HEADS * WIDTH))
            lse_bytes.extend(NEG_INF * HEADS)
            continue
        slots = tuple(
            prepared.indices[prepared.indptr[b] + t // PAGE] * PAGE + t % PAGE for t in range(n)
        )
        for h in range(HEADS):
            off = (b * HEADS + h) * WIDTH
            query = q[off : off + WIDTH]
            offsets = tuple((slot * KV_HEADS + (h >= 8)) * WIDTH for slot in slots)
            u = tuple(
                math.fsum(x * y for x, y in zip(query, k[i : i + WIDTH], strict=True))
                * prepared.scale
                for i in offsets
            )
            peak = max(u)
            weights = tuple(math.exp(x - peak) for x in u)
            mass = math.fsum(weights)
            out_bytes.extend(
                struct.pack(
                    f"<{WIDTH}H",
                    *(
                        bf16(
                            math.fsum(w * v[i + d] for w, i in zip(weights, offsets, strict=True))
                            / mass
                        )
                        for d in range(WIDTH)
                    ),
                )
            )
            lse_bytes.extend(struct.pack("<f", (peak + math.log(mass)) / math.log(2)))
    return bytes(out_bytes), bytes(lse_bytes)


def compare(prepared: Prepared, outputs: tuple[bytes, bytes]) -> dict[str, int]:
    expected = oracle(prepared)
    report = {}
    for role, raw, want, width, unit, decode, sentinel in zip(
        ("O", "L"),
        outputs,
        expected,
        (HEADS * WIDTH, HEADS),
        (2, 4),
        (unpack_bf16, lambda x: struct.unpack(f"<{len(x) // 4}f", x)),
        (ZERO_BF16, NEG_INF),
        strict=True,
    ):
        if type(raw) is not bytes or len(raw) != len(want):
            raise ValueError(f"{role}: wrong output byte count")
        fails = 0
        stride = width * unit
        for b, n in enumerate(prepared.token_counts()):
            lo, hi = b * stride, (b + 1) * stride
            if not n:
                if raw[lo:hi] != sentinel * width:
                    raise ValueError(f"{role}: bad empty sentinel")
                continue
            for actual, target in zip(decode(raw[lo:hi]), decode(want[lo:hi]), strict=True):
                if not math.isfinite(actual):
                    raise ValueError(f"{role}: nonfinite or untouched output")
                error = abs(actual - target)
                fails += error > 0.01 and error / (abs(target) + 1e-8) > 0.01
        report[role] = fails
    return report
