"""Existing OOLONG auxiliary score; no dataset or training imports."""

import ast
from datetime import datetime
from typing import Any

import dateutil.parser

COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")


def _find_comparison_phrase(output: str) -> str | None:
    out_low = output.lower()
    hits = [(out_low.rfind(p), p) for p in COMPARISON_PHRASES if p in out_low]
    return max(hits)[1] if hits else None


def _attempt_answer_parse(answer: str) -> tuple[str, str]:
    cmp = _find_comparison_phrase(answer)
    if cmp is not None:
        return cmp, "high"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, "low"
        return answer.split()[-1], "low"
    cand = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    if len(cand) < 20:
        return cand, "vhigh"
    return cand, "med"


def synth_score(datapoint: dict, output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    trimmed, _ = _attempt_answer_parse(output)
    gold_s = str(gold)

    if str(trimmed) == gold_s:
        return 1.0
    if str(trimmed).lower() == gold_s.lower():
        return 1.0

    atype = datapoint.get("answer_type", "")
    if atype == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except Exception:
            return 0.0
    if atype == "ANSWER_TYPE.DATE":
        try:
            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except Exception:
            return 0.0

    if gold_s and gold_s.lower() not in [p.lower() for p in COMPARISON_PHRASES]:
        if gold_s.lower() in output.lower():
            return 1.0

    return 0.0
