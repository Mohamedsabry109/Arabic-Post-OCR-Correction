#!/usr/bin/env python3
"""Run all pipeline phases sequentially.

Executes the local portions of every phase in order:
  - Phase 1:  fully local (no LLM)
  - Phases 2-6: export stage (local), then analyze/validate/summarize (local)

Inference (LLM on GPU) is a separate step that must be run on Kaggle/Colab.
After running --mode export, follow the prompts or see HOW_TO_RUN.md for
the inference commands, then run --mode analyze.

Usage
-----
    # Export everything (prep all inference inputs)
    python pipelines/run_all.py --mode export

    # Analyze everything (after inference files are in place)
    python pipelines/run_all.py --mode analyze

    # Full local run (export + infer locally + analyze) — requires local GPU
    python pipelines/run_all.py --mode full

    # Smoke test export (50 samples, KHATT-train only)
    python pipelines/run_all.py --mode export --limit 50 --datasets KHATT-train

    # Specific phases only
    python pipelines/run_all.py --mode export --phases 2 3 4a
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

# Each entry: (phase_key, script, modes_needed_for_export, modes_needed_for_analyze)
# mode strings are passed to the underlying script as --mode <x>
# Special values: None = no --mode flag (script handles it directly)

_PHASE_STEPS: dict[str, list[dict]] = {
    "1": [
        {"script": "pipelines/run_phase1.py", "mode": None, "stage": "local"},
    ],
    "2": [
        {"script": "pipelines/run_phase2.py", "mode": "export",  "stage": "export"},
        {"script": "pipelines/run_phase2.py", "mode": "analyze", "stage": "analyze"},
    ],
    "3": [
        {"script": "pipelines/run_phase3.py", "mode": "export",  "stage": "export"},
        {"script": "pipelines/run_phase3.py", "mode": "analyze", "stage": "analyze"},
    ],
    "4a": [
        {"script": "pipelines/run_phase4.py", "mode": "export",  "stage": "export",
         "extra": ["--sub-phase", "4a"]},
        {"script": "pipelines/run_phase4.py", "mode": "analyze", "stage": "analyze",
         "extra": ["--sub-phase", "4a"]},
    ],
    "4b": [
        {"script": "pipelines/run_phase4.py", "mode": "export",  "stage": "export",
         "extra": ["--sub-phase", "4b"]},
        {"script": "pipelines/run_phase4.py", "mode": "analyze", "stage": "analyze",
         "extra": ["--sub-phase", "4b"]},
    ],
    "4c": [
        {"script": "pipelines/run_phase4.py", "mode": "validate", "stage": "analyze",
         "extra": ["--sub-phase", "4c"]},
    ],
    "5": [
        {"script": "pipelines/run_phase5.py", "mode": "build",   "stage": "local"},
        {"script": "pipelines/run_phase5.py", "mode": "export",  "stage": "export"},
        {"script": "pipelines/run_phase5.py", "mode": "analyze", "stage": "analyze"},
    ],
    "6": [
        {"script": "pipelines/run_phase6.py", "mode": "export",   "stage": "export",
         "extra": ["--combo", "all"]},
        {"script": "pipelines/run_phase6.py", "mode": "analyze",  "stage": "analyze",
         "extra": ["--combo", "all"]},
        {"script": "pipelines/run_phase6.py", "mode": "validate", "stage": "analyze",
         "extra": ["--combo", "full_system"]},
        {"script": "pipelines/run_phase6.py", "mode": "summarize","stage": "analyze"},
    ],
}

_ALL_PHASES = ["1", "2", "3", "4a", "4b", "4c", "5", "6"]

# Inference input/output paths per phase (for --mode full)
_INFERENCE_IO: dict[str, tuple[str, str]] = {
    "2":  ("results/phase2/inference_input.jsonl",  "results/phase2/corrections.jsonl"),
    "3":  ("results/phase3/inference_input.jsonl",  "results/phase3/corrections.jsonl"),
    "4a": ("results/phase4a/inference_input.jsonl", "results/phase4a/corrections.jsonl"),
    "4b": ("results/phase4b/inference_input.jsonl", "results/phase4b/corrections.jsonl"),
    "5":  ("results/phase5/inference_input.jsonl",  "results/phase5/corrections.jsonl"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Arabic OCR correction pipeline phases sequentially."
    )
    parser.add_argument(
        "--mode",
        choices=["export", "analyze", "full"],
        default="export",
        help=(
            "export: run all export stages (prep inference inputs). "
            "analyze: run all analyze/validate/summarize stages. "
            "full: export + infer locally + analyze (requires local GPU)."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        metavar="PHASE",
        help=(
            "Phases to run (default: all). "
            "Options: 1 2 3 4a 4b 4c 5 6. "
            "Example: --phases 2 3 4a"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Process only these dataset keys (passed through to each pipeline).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset for testing (passed through to each pipeline).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run ignoring cached results (passed through to each pipeline).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Config YAML path (passed through to each pipeline).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _build_cmd(
    step: dict,
    args: argparse.Namespace,
) -> list[str]:
    """Build the subprocess command for a single pipeline step."""
    cmd = [_PYTHON, str(_PROJECT_ROOT / step["script"])]

    if step.get("mode") is not None:
        cmd += ["--mode", step["mode"]]

    if step.get("extra"):
        cmd += step["extra"]

    if args.datasets:
        cmd += ["--datasets"] + args.datasets

    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    if args.force:
        cmd += ["--force"]

    # Phase 5 build has --max-sentences; pass limit as that flag too
    if step["script"] == "pipelines/run_phase5.py" and step.get("mode") == "build":
        if args.limit is not None:
            cmd += ["--max-sentences", str(args.limit * 5)]

    return cmd


def _run_step(step: dict, args: argparse.Namespace, phase_key: str) -> bool:
    """Run one step. Returns True on success."""
    cmd = _build_cmd(step, args)
    label = f"Phase {phase_key.upper()} | {step['script']} --mode {step.get('mode', 'N/A')}"
    logger.info("=" * 60)
    logger.info("Running: %s", label)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("FAILED: %s (exit code %d)", label, result.returncode)
        return False

    logger.info("OK: %s", label)
    return True


def _run_inference(phase_key: str, args: argparse.Namespace) -> bool:
    """Run scripts/infer.py for a given phase (--mode full only)."""
    if phase_key not in _INFERENCE_IO:
        return True  # Phase 1, 4c, 6 combos handled separately

    input_path, output_path = _INFERENCE_IO[phase_key]
    cmd = [
        _PYTHON, str(_PROJECT_ROOT / "scripts/infer.py"),
        "--input",  input_path,
        "--output", output_path,
    ]
    if args.datasets:
        cmd += ["--datasets"] + args.datasets
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.force:
        cmd += ["--force"]

    label = f"Phase {phase_key.upper()} | Inference"
    logger.info("=" * 60)
    logger.info("Running: %s", label)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 60)

    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        logger.error("FAILED: %s (exit code %d)", label, result.returncode)
        return False

    logger.info("OK: %s", label)
    return True


def _run_phase6_inference_full(args: argparse.Namespace) -> bool:
    """Run infer.py for each Phase 6 inference combo (--mode full only)."""
    inference_combos = [
        "pair_conf_rules", "pair_conf_fewshot", "pair_conf_rag",
        "pair_rules_fewshot", "full_prompt",
        "abl_no_confusion", "abl_no_rules", "abl_no_fewshot", "abl_no_rag",
    ]
    for combo in inference_combos:
        input_path  = f"results/phase6/{combo}/inference_input.jsonl"
        output_path = f"results/phase6/{combo}/corrections.jsonl"
        cmd = [
            _PYTHON, str(_PROJECT_ROOT / "scripts/infer.py"),
            "--input",  input_path,
            "--output", output_path,
        ]
        if args.datasets:
            cmd += ["--datasets"] + args.datasets
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.force:
            cmd += ["--force"]

        label = f"Phase 6 | Inference | {combo}"
        logger.info("=" * 60)
        logger.info("Running: %s", label)
        logger.info("=" * 60)
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
        if result.returncode != 0:
            logger.error("FAILED: %s (exit code %d)", label, result.returncode)
            return False
        logger.info("OK: %s", label)
    return True


def main() -> None:
    args = parse_args()

    phases = args.phases if args.phases else _ALL_PHASES
    # Normalise to lowercase
    phases = [p.lower() for p in phases]

    unknown = [p for p in phases if p not in _ALL_PHASES]
    if unknown:
        logger.error("Unknown phase(s): %s. Valid options: %s", unknown, _ALL_PHASES)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("run_all.py | mode=%s | phases=%s", args.mode, phases)
    if args.limit:
        logger.info("limit=%d per dataset", args.limit)
    if args.datasets:
        logger.info("datasets=%s", args.datasets)
    logger.info("=" * 60)

    failed: list[str] = []

    for phase_key in phases:
        steps = _PHASE_STEPS[phase_key]

        for step in steps:
            stage = step["stage"]

            # Decide whether to run this step based on mode
            if args.mode == "export":
                if stage not in ("export", "local"):
                    continue
            elif args.mode == "analyze":
                if stage not in ("analyze", "local"):
                    # Phase 5 build is 'local' — skip on analyze-only pass
                    # (index should already exist)
                    if stage == "local" and step.get("mode") == "build":
                        continue
                    elif stage != "analyze":
                        continue
            # mode == "full": run everything

            if not _run_step(step, args, phase_key):
                failed.append(f"Phase {phase_key.upper()} | {step.get('mode', 'N/A')}")
                logger.error("Stopping due to failure in Phase %s.", phase_key.upper())
                break

            # Inject inference step between export and analyze in --mode full
            if args.mode == "full" and stage == "export":
                if phase_key == "6":
                    if not _run_phase6_inference_full(args):
                        failed.append(f"Phase 6 | Inference")
                        break
                else:
                    if not _run_inference(phase_key, args):
                        failed.append(f"Phase {phase_key.upper()} | Inference")
                        break

        if failed:
            break

    logger.info("=" * 60)
    if failed:
        logger.error("run_all.py finished with failures:")
        for f in failed:
            logger.error("  FAILED: %s", f)
        sys.exit(1)
    else:
        logger.info("run_all.py complete — all steps succeeded.")
        if args.mode == "export":
            logger.info("")
            logger.info("Next step: run inference on Kaggle/Colab for each phase,")
            logger.info("then run: python pipelines/run_all.py --mode analyze")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
