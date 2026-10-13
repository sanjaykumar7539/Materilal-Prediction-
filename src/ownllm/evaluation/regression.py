"""Simple exact / normalized match regression harness."""
from __future__ import annotations

from typing import Any, Callable


def exact_match(pred: str, truth: str) -> bool:
    return pred.strip() == truth.strip()


def normalized_match(pred: str, truth: str) -> bool:
    return " ".join(pred.lower().split()) == " ".join(truth.lower().split())


def run_cases(generate: Callable[[str], str], cases: list[dict]) -> dict[str, Any]:
    out = []
    for case in cases:
        pred = generate(case["prompt"])
        use_norm = case.get("normalized", True)
        ok = normalized_match(pred, case["expected"]) if use_norm else exact_match(pred, case["expected"])
        out.append({**case, "prediction": pred, "pass": ok})
    return {
        "passed": sum(1 for x in out if x["pass"]),
        "total": len(out),
        "cases": out,
    }
