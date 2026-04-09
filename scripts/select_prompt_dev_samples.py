#!/usr/bin/env python3
"""Select a small representative sample set for fast prompt iteration.

Unlike select_test_samples.py (which stratifies by OCR CER difficulty),
this script stratifies by **LLM correction outcome** using existing
Phase 2 training results.  This ensures we test:

  - FALSE POSITIVES: OCR was correct/near-correct but LLM damaged it
  - UNFIXED: OCR had errors that LLM left unchanged
  - IMPROVED: OCR had errors that LLM partially fixed (don't regress)
  - PERFECT: LLM nailed it (don't regress)
  - RUNAWAY: OCR hallucination loops (test truncation handling)

The output is a JSON file with sample_ids that can be passed to
``scripts/infer.py --sample-list`` or ``craft_prompt.py --sample-list``.

Usage:
    python scripts/select_prompt_dev_samples.py
    python scripts/select_prompt_dev_samples.py --n 300 --include-khatt
    python scripts/select_prompt_dev_samples.py --corrections results/phase2/corrections.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.metrics import calculate_cer
from src.data.text_utils import normalise_arabic

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CORRECTIONS = _PROJECT_ROOT / "results" / "phase2" / "corrections.jsonl"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "prompt_dev_samples.json"
_SEED = 42

# Bucket target sizes (total ~300 by default)
_TARGETS = {
    "false_positive":  80,   # OCR CER < 2% but LLM made it worse
    "degraded_real":   40,   # OCR had real errors, LLM made it worse
    "unfixed_hard":    40,   # OCR CER > 5%, LLM didn't help
    "improved":        40,   # LLM improved CER (don't regress)
    "perfect":         40,   # LLM hit CER=0 (don't regress)
    "runaway":         20,   # OCR > 5x GT length (hallucination loops)
    "correct_ocr":     40,   # OCR was perfect, LLM should leave alone
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select prompt-iteration dev samples stratified by LLM outcome."
    )
    p.add_argument(
        "--corrections", type=Path, default=_DEFAULT_CORRECTIONS,
        help="Path to Phase 2 corrections.jsonl.",
    )
    p.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Total samples (scales bucket sizes proportionally).",
    )
    p.add_argument(
        "--include-khatt", action="store_true", dest="include_khatt",
        help="Include KHATT-train samples.",
    )
    p.add_argument("--seed", type=int, default=_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Scale targets if --n provided
    targets = dict(_TARGETS)
    if args.n:
        base_total = sum(_TARGETS.values())
        scale = args.n / base_total
        targets = {k: max(1, round(v * scale)) for k, v in _TARGETS.items()}

    total_target = sum(targets.values())
    print(f"Target: {total_target} samples")
    for k, v in targets.items():
        print(f"  {k}: {v}")

    # Load corrections
    if not args.corrections.exists():
        print(f"ERROR: {args.corrections} not found")
        sys.exit(1)

    records: list[dict] = []
    with open(args.corrections, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("gt_text") or not r.get("ocr_text"):
                continue
            if not args.include_khatt and "KHATT" in r.get("dataset", ""):
                continue
            records.append(r)

    print(f"Loaded {len(records)} records")

    # Classify each record
    buckets: dict[str, list[dict]] = {k: [] for k in targets}

    for r in records:
        gt_text = r["gt_text"]
        ocr_text = r["ocr_text"]
        llm_text = r.get("corrected_text", ocr_text)
        dataset = r.get("dataset", "")

        gt_norm = normalise_arabic(gt_text, remove_diacritics=True)
        ocr_norm = normalise_arabic(ocr_text, remove_diacritics=True)

        # Skip near-empty
        if len(gt_norm) < 5:
            continue

        cer_ocr = calculate_cer(gt_text, ocr_text, strip_diacritics=True)
        cer_llm = calculate_cer(gt_text, llm_text, strip_diacritics=True)
        delta = cer_llm - cer_ocr

        ocr_ratio = len(ocr_norm) / max(len(gt_norm), 1)

        entry = {
            "sample_id": r["sample_id"],
            "dataset": dataset,
            "cer_ocr": round(cer_ocr, 6),
            "cer_llm": round(cer_llm, 6),
            "cer_delta": round(delta, 6),
            "ocr_ratio": round(ocr_ratio, 2),
            "gt_chars": len(gt_norm),
        }

        # Classify into buckets (priority order)
        if ocr_ratio > 5.0:
            buckets["runaway"].append(entry)
        elif cer_ocr == 0.0 and cer_llm == 0.0:
            buckets["correct_ocr"].append(entry)
        elif cer_ocr < 0.02 and delta > 0.005:
            buckets["false_positive"].append(entry)
        elif cer_ocr >= 0.02 and delta > 0.01:
            buckets["degraded_real"].append(entry)
        elif abs(delta) <= 0.005 and cer_ocr > 0.05:
            buckets["unfixed_hard"].append(entry)
        elif cer_llm == 0.0 and cer_ocr > 0:
            buckets["perfect"].append(entry)
        elif delta < -0.005:
            buckets["improved"].append(entry)

    print(f"\nBucket pools:")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)} available (target: {targets[k]})")

    # Select from each bucket with dataset diversity
    selected_by_bucket: dict[str, list[dict]] = {}
    for bucket_name, target_n in targets.items():
        pool = buckets[bucket_name]
        rng.shuffle(pool)

        # First pass: one per dataset for diversity
        picked: list[dict] = []
        seen_datasets: set[str] = set()
        for entry in pool:
            if len(picked) >= target_n:
                break
            ds = entry["dataset"]
            if ds not in seen_datasets:
                picked.append(entry)
                seen_datasets.add(ds)

        # Second pass: fill remaining
        for entry in pool:
            if len(picked) >= target_n:
                break
            if entry not in picked:
                picked.append(entry)

        # Sort by severity (worst first for failure buckets)
        if bucket_name in ("false_positive", "degraded_real"):
            picked.sort(key=lambda x: x["cer_delta"], reverse=True)
        elif bucket_name == "unfixed_hard":
            picked.sort(key=lambda x: x["cer_ocr"], reverse=True)

        selected_by_bucket[bucket_name] = picked

    # Combine
    selected: list[dict] = []
    for bucket_name in targets:
        for s in selected_by_bucket[bucket_name]:
            s["category"] = bucket_name
            selected.append(s)

    # Summary
    ds_counts: Counter = Counter()
    for s in selected:
        ds_counts[s["dataset"]] += 1

    print(f"\nSelected {len(selected)} samples:")
    for bucket_name in targets:
        n = len(selected_by_bucket[bucket_name])
        print(f"  {bucket_name}: {n}")
    print(f"\nBy dataset:")
    for ds, count in sorted(ds_counts.items()):
        print(f"  {ds}: {count}")

    # Write output
    output = {
        "meta": {
            "description": "Prompt-iteration dev samples stratified by LLM correction outcome",
            "total_selected": len(selected),
            "by_category": {k: len(v) for k, v in selected_by_bucket.items()},
            "seed": args.seed,
            "source": str(args.corrections),
            "by_dataset": dict(ds_counts),
        },
        "sample_ids": [s["sample_id"] for s in selected],
        "samples": selected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output}")
    print(f"\nUsage:")
    print(f"  python scripts/infer.py --sample-list {args.output} ...")


if __name__ == "__main__":
    main()
