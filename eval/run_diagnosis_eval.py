"""
Evaluate Medi-Match DiagnosisEngine top-1 / top-3 / MRR on labeled symptom cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.diagnosis_engine import create_diagnosis_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alias_set(gold_id: str, aliases: Sequence[str]) -> Set[str]:
    names = {normalize(gold_id.replace("_", " ")), normalize(gold_id)}
    for alias in aliases:
        names.add(normalize(alias))
    return {n for n in names if n}


def prediction_matches(pred_name: str, gold_names: Set[str]) -> bool:
    pred = normalize(pred_name)
    if not pred:
        return False
    if pred in gold_names:
        return True
    for gold in gold_names:
        if gold and (gold in pred or pred in gold):
            # Avoid ultra-short accidental substring matches
            if min(len(gold), len(pred)) >= 4:
                return True
    return False


def rank_of_gold(predictions: Sequence[str], gold_names: Set[str]) -> Optional[int]:
    for idx, name in enumerate(predictions, start=1):
        if prediction_matches(name, gold_names):
            return idx
    return None


def run_case(engine, case: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    result = engine.analyze_symptoms(
        primary_symptom=case["primary_symptom"],
        secondary_symptoms=case.get("secondary_symptoms") or [],
        patient_info=case.get("patient_info"),
        medical_history=case.get("medical_history") or [],
        duration_days=case.get("duration_days"),
    )
    elapsed = time.time() - started

    if result.get("error"):
        return {
            "id": case["id"],
            "gold_id": case["gold_id"],
            "error": result.get("error"),
            "details": result.get("details"),
            "elapsed_sec": round(elapsed, 2),
            "predictions": [],
            "rank": None,
            "top1": False,
            "top3": False,
            "top5": False,
        }

    diagnoses = result.get("diagnoses") or []
    predictions = [d.get("name", "") for d in diagnoses if d.get("name")]
    gold_names = alias_set(case["gold_id"], case.get("gold_aliases") or [])
    rank = rank_of_gold(predictions, gold_names)

    return {
        "id": case["id"],
        "gold_id": case["gold_id"],
        "elapsed_sec": round(elapsed, 2),
        "predictions": predictions,
        "confidences": [d.get("confidence") for d in diagnoses],
        "rank": rank,
        "top1": rank == 1,
        "top3": rank is not None and rank <= 3,
        "top5": rank is not None and rank <= 5,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in results if not r.get("error")]
    n = len(scored)
    if n == 0:
        return {"n_cases": 0, "n_errors": len(results)}

    top1 = sum(1 for r in scored if r["top1"])
    top3 = sum(1 for r in scored if r["top3"])
    top5 = sum(1 for r in scored if r["top5"])
    mrr_vals = [(1.0 / r["rank"]) if r["rank"] else 0.0 for r in scored]
    avg_latency = sum(r["elapsed_sec"] for r in scored) / n

    return {
        "n_cases": n,
        "n_errors": len(results) - n,
        "top1_accuracy": round(100.0 * top1 / n, 1),
        "top3_hit_rate": round(100.0 * top3 / n, 1),
        "top5_hit_rate": round(100.0 * top5 / n, 1),
        "mrr": round(sum(mrr_vals) / n, 3),
        "avg_latency_sec": round(avg_latency, 1),
        "top1_count": top1,
        "top3_count": top3,
        "top5_count": top5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Medi-Match diagnosis accuracy eval")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
        help="Path to labeled cases JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases to run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results.json"),
        help="Where to write full results JSON",
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.limit is not None:
        cases = cases[: args.limit]

    logger.info("Loading DiagnosisEngine...")
    engine = create_diagnosis_engine()
    logger.info("Running %d cases...", len(cases))

    results: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        logger.info("[%d/%d] %s (gold=%s)", i, len(cases), case["id"], case["gold_id"])
        row = run_case(engine, case)
        results.append(row)
        logger.info(
            "  -> top1=%s top3=%s rank=%s preds=%s (%.1fs)",
            row.get("top1"),
            row.get("top3"),
            row.get("rank"),
            row.get("predictions"),
            row.get("elapsed_sec", 0),
        )

    metrics = summarize(results)
    payload = {"metrics": metrics, "results": results}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n===== DIAGNOSIS EVAL SUMMARY =====")
    print(json.dumps(metrics, indent=2))
    print(f"\nFull results written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
