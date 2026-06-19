"""Smart runaway corrector for LLM post-correction outputs.

Three runaway patterns observed in practice:

  1. Tatweel flood   — OCR produced blank / noise; LLM echoes it:
       "ـ ـ ـ ـ ـ ـ ـ ـ ـ ..."   →  "" (empty)

  2. Phrase loop     — LLM starts correctly then enters infinite loop:
       "أخبرنا مالك بن … صلى الله عليه وسلم صلى الله عليه وسلم صلى…"
       →  "أخبرنا مالك بن … صلى الله عليه وسلم"

  3. Word/short loop — LLM repeats a single word:
       "المجمع المجمع المجمع المجمع …"  →  "المجمع"

Detection: len(corrected) > ratio_threshold × len(gt)   (default ratio = 3.0)

Usage
-----
    from src.analysis.runaway_corrector import fix_text, apply_to_records

    # Fix a single corrected text
    fixed = fix_text(corrected_text, gt_text)

    # Apply to a list of correction dicts in-place (adds 'corrected_text_fixed')
    records = apply_to_records(records, ratio_threshold=3.0)
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum phrase length to consider when scanning for loops
_MIN_PHRASE = 6
# Maximum phrase length (loops are usually religious/formulaic, ≤ 60 chars)
_MAX_PHRASE = 80
# How many times a phrase must appear to count as a loop
_MIN_REPS   = 2
# Search for loop start only within this many × ref_len chars
_SEARCH_MULTIPLIER = 4

# Arabic Unicode range (main block)
_ARABIC_RE = re.compile(r'[؀-ۿ]')
# Tatweel / kashida character
_TATWEEL_RE = re.compile(r'(ـ[\s]*){3,}')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_runaway(text: str, ref_len: int, ratio: float) -> bool:
    return len(text) > max(ref_len * ratio, 50)


def _collapse_tatweel(text: str) -> str:
    """Collapse consecutive tatweel sequences to a single ـ."""
    return _TATWEEL_RE.sub('ـ', text).strip()


def _find_loop_start(text: str, ref_len: int) -> Optional[int]:
    """
    Return the index where the first phrase-loop begins, or None.

    Scans for a phrase of length _MIN_PHRASE.._MAX_PHRASE that appears
    twice in quick succession within the first SEARCH_MULTIPLIER × ref_len
    characters.  The phrase must contain at least one Arabic character.
    """
    search_zone_end = min(len(text), max(ref_len * _SEARCH_MULTIPLIER, 300))
    zone = text[:search_zone_end]
    n = len(zone)

    for phrase_len in range(_MIN_PHRASE, min(_MAX_PHRASE, n // 2) + 1):
        # Only scan start positions up to 2× ref_len (loop starts early)
        scan_end = min(n - phrase_len * 2, ref_len * 2)
        for start in range(0, max(scan_end, 0)):
            phrase = zone[start : start + phrase_len]

            # Phrase must carry meaningful Arabic content
            if not _ARABIC_RE.search(phrase):
                continue

            second = zone.find(phrase, start + phrase_len)
            if second == -1:
                continue

            # Tight loop: second occurrence starts immediately after first ends
            # (allow gap ≤ max(5, phrase_len * 0.3) to handle space separators)
            max_gap = max(5, int(phrase_len * 0.3))
            if second <= start + phrase_len + max_gap:
                return second  # cut right before the second occurrence

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fix_text(
    corrected_text: str,
    gt_text: str = "",
    ocr_text: str = "",
    ratio_threshold: float = 3.0,
) -> str:
    """Return a cleaned version of *corrected_text* if it is a runaway.

    If *corrected_text* is not a runaway (length ≤ ratio_threshold × |gt|),
    it is returned unchanged.

    Correction cascade:
      1. Tatweel-flood collapse → if result is tiny, return ""
      2. Sliding-window phrase-loop detection → truncate at loop start
      3. Hard truncate to 2 × |gt| as last resort

    Parameters
    ----------
    corrected_text : str
        Output from the LLM correction step.
    gt_text : str
        Ground-truth text (used to estimate expected length).
    ocr_text : str
        Original OCR text (used as fallback reference length).
    ratio_threshold : float
        corrected / gt ratio above which the text is considered a runaway.

    Returns
    -------
    str
        Fixed text (possibly empty if the original was tatweel noise).
    """
    if not corrected_text:
        return corrected_text

    # Estimate reference length
    if gt_text:
        ref_len = max(len(gt_text), 10)
    elif ocr_text:
        ref_len = max(len(ocr_text), 10)
    else:
        ref_len = 50

    # Not a runaway — return as-is
    if not _is_runaway(corrected_text, ref_len, ratio_threshold):
        return corrected_text

    # ── Strategy 1: tatweel flood ──────────────────────────────────────────
    collapsed = _collapse_tatweel(corrected_text)
    # After collapsing, if the text shrank dramatically it was tatweel noise
    if len(collapsed) < len(corrected_text) * 0.15:
        # If the remaining content is tiny / uninformative → empty string
        arabic_chars = len(_ARABIC_RE.findall(collapsed))
        return collapsed if arabic_chars >= 3 else ""

    # Work with the collapsed version from here
    work = collapsed

    # ── Strategy 2: phrase-loop detection ─────────────────────────────────
    loop_pos = _find_loop_start(work, ref_len)
    if loop_pos is not None and loop_pos > 0:
        prefix = work[:loop_pos].rstrip()
        # Only keep the prefix if it's non-trivial
        if len(prefix) >= 3 and _ARABIC_RE.search(prefix):
            return prefix

    # ── Strategy 3: hard truncate ──────────────────────────────────────────
    limit = int(ref_len * 2)
    return work[:limit].rstrip()


def apply_to_records(
    records: list[dict],
    ratio_threshold: float = 3.0,
    field_in:  str = "corrected_text",
    field_out: str = "corrected_text_fixed",
) -> tuple[list[dict], dict]:
    """Apply *fix_text* to every record and store result in *field_out*.

    Parameters
    ----------
    records : list[dict]
        Correction records (each must have 'corrected_text', 'gt_text').
    ratio_threshold : float
        Passed to fix_text.
    field_in : str
        Source field name (default "corrected_text").
    field_out : str
        Destination field name (default "corrected_text_fixed").

    Returns
    -------
    (records_with_field_out, stats)
        stats keys: total, fixed, unchanged, emptied
    """
    stats = {"total": 0, "fixed": 0, "unchanged": 0, "emptied": 0}

    for r in records:
        corr = r.get(field_in, "")
        gt   = r.get("gt_text",  "")
        ocr  = r.get("ocr_text", "")

        fixed = fix_text(corr, gt_text=gt, ocr_text=ocr,
                         ratio_threshold=ratio_threshold)

        r[field_out] = fixed
        stats["total"] += 1

        if fixed == corr:
            stats["unchanged"] += 1
        elif fixed == "":
            stats["emptied"] += 1
        else:
            stats["fixed"] += 1

    return records, stats


# ---------------------------------------------------------------------------
# CLI convenience — apply to a corrections JSONL and print metric delta
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse, json, sys, io, statistics
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.analysis.metrics import calculate_cer, calculate_wer
    from src.data.text_utils import normalise_arabic

    ap = argparse.ArgumentParser(
        description="Apply runaway corrector to a corrections JSONL and report CER delta."
    )
    ap.add_argument("input",  type=Path, help="corrections.jsonl")
    ap.add_argument("--ratio", type=float, default=3.0,
                    help="Runaway ratio threshold (default 3.0)")
    ap.add_argument("--output", type=Path, default=None,
                    help="If given, write fixed JSONL to this path.")
    args = ap.parse_args()

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records, stats = apply_to_records(records, ratio_threshold=args.ratio)

    print(f"Records : {stats['total']}")
    print(f"Fixed   : {stats['fixed']}  (loop/tatweel corrected)")
    print(f"Emptied : {stats['emptied']}  (pure tatweel → empty)")
    print(f"Unchanged: {stats['unchanged']}")
    print()

    # Compute CER before / after per dataset
    by_ds: dict[str, list] = {}
    for r in records:
        by_ds.setdefault(r.get("dataset","?"), []).append(r)

    print(f"{'Dataset':<30}  {'CER_orig':>9}  {'CER_fixed':>9}  {'Delta':>8}  {'N_fixed':>8}")
    print("-" * 72)
    for ds, recs in sorted(by_ds.items()):
        orig_cers, fix_cers = [], []
        n_fixed = 0
        for r in recs:
            gt = r.get("gt_text","")
            c_orig = r.get("corrected_text","")
            c_fix  = r.get("corrected_text_fixed", c_orig)
            orig_cers.append(calculate_cer(gt, c_orig, strip_diacritics=True))
            fix_cers.append(calculate_cer(gt, c_fix,  strip_diacritics=True))
            if c_fix != c_orig:
                n_fixed += 1
        mean_orig = statistics.mean(orig_cers)
        mean_fix  = statistics.mean(fix_cers)
        delta     = mean_orig - mean_fix
        print(f"{ds:<30}  {mean_orig*100:>8.2f}%  {mean_fix*100:>8.2f}%  {delta*100:>+7.2f}%  {n_fixed:>8}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nFixed JSONL written to: {args.output}")


if __name__ == "__main__":
    _cli()
