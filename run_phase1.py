#!/usr/bin/env python3
"""
Phase 1 Pipeline: LLM Baseline for Arabic Post-OCR Correction.

This script runs the complete Phase 1 pipeline:
1. Load PATS-A01 and KHATT datasets
2. Calculate baseline metrics (OCR vs Ground Truth)
3. Run zero-shot LLM correction
4. Calculate Phase 1 metrics (Corrected vs Ground Truth)
5. Save results and generate report

Usage:
    python run_phase1.py
    python run_phase1.py --config config.yaml
    python run_phase1.py --datasets PATS-A01 KHATT --limit 100

Example:
    # Run with default settings
    python run_phase1.py

    # Run with specific datasets and sample limit
    python run_phase1.py --datasets PATS-A01 --limit 50

    # Run with custom config
    python run_phase1.py --config my_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml
from tqdm import tqdm

from src.data_loader import DataLoader, DataLoaderError
from src.metrics import calculate_metrics, calculate_improvement, format_metrics
from src.llm_corrector import LLMCorrector, LLMCorrectorError
from src.utils import format_sample_correction, clean_text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('phase1.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'data': {
            'ground_truth_base': 'data/Original',
            'predictions_base': 'data/Predictions',
            'datasets': ['PATS-A01', 'KHATT'],
        },
        'model': {
            'name': 'Qwen/Qwen2.5-3B-Instruct',
            'temperature': 0.1,
            'max_new_tokens': 1024,
            'load_in_4bit': False,
            'load_in_8bit': False,
        },
        'output': {
            'results_dir': 'results/phase1',
            'num_samples_to_show': 20,
        },
        'processing': {
            'limit_per_dataset': None,  # None for all samples
        }
    }


def run_baseline_evaluation(
    data: Dict[str, List[Tuple[str, str]]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate baseline metrics (OCR vs Ground Truth).

    Args:
        data: Dictionary mapping dataset names to (ocr, ground_truth) pairs.

    Returns:
        Dictionary mapping dataset names to metrics.
    """
    logger.info("Calculating baseline metrics (OCR vs Ground Truth)...")
    baseline_metrics = {}

    for dataset_name, pairs in data.items():
        if not pairs:
            logger.warning(f"No data for dataset: {dataset_name}")
            continue

        ocr_texts = [p[0] for p in pairs]
        ground_truths = [p[1] for p in pairs]

        metrics = calculate_metrics(ocr_texts, ground_truths)
        baseline_metrics[dataset_name] = metrics

        logger.info(
            f"Baseline {dataset_name}: CER={metrics['cer']:.2%}, "
            f"WER={metrics['wer']:.2%}"
        )

    return baseline_metrics


def run_llm_correction(
    corrector: LLMCorrector,
    data: Dict[str, List[Tuple[str, str]]],
    progress: bool = True
) -> Dict[str, List[str]]:
    """
    Run LLM correction on all datasets.

    Args:
        corrector: LLMCorrector instance.
        data: Dictionary mapping dataset names to (ocr, ground_truth) pairs.
        progress: Show progress bars.

    Returns:
        Dictionary mapping dataset names to corrected texts.
    """
    logger.info("Running LLM correction...")
    corrections = {}

    for dataset_name, pairs in data.items():
        if not pairs:
            corrections[dataset_name] = []
            continue

        logger.info(f"Correcting {dataset_name} ({len(pairs)} samples)...")
        ocr_texts = [p[0] for p in pairs]

        corrected_texts = []
        iterator = tqdm(ocr_texts, desc=f"Correcting {dataset_name}") if progress else ocr_texts

        for ocr_text in iterator:
            try:
                corrected = corrector.correct(ocr_text)
                corrected_texts.append(corrected)
            except LLMCorrectorError as e:
                logger.warning(f"Correction failed, using original: {e}")
                corrected_texts.append(ocr_text)

        corrections[dataset_name] = corrected_texts
        logger.info(f"Completed {dataset_name}: {len(corrected_texts)} corrections")

    return corrections


def run_phase1_evaluation(
    corrections: Dict[str, List[str]],
    data: Dict[str, List[Tuple[str, str]]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Phase 1 metrics (Corrected vs Ground Truth).

    Args:
        corrections: Dictionary mapping dataset names to corrected texts.
        data: Dictionary mapping dataset names to (ocr, ground_truth) pairs.

    Returns:
        Dictionary mapping dataset names to metrics.
    """
    logger.info("Calculating Phase 1 metrics (Corrected vs Ground Truth)...")
    phase1_metrics = {}

    for dataset_name, corrected_texts in corrections.items():
        pairs = data.get(dataset_name, [])
        if not pairs or not corrected_texts:
            continue

        ground_truths = [p[1] for p in pairs]
        metrics = calculate_metrics(corrected_texts, ground_truths)
        phase1_metrics[dataset_name] = metrics

        logger.info(
            f"Phase 1 {dataset_name}: CER={metrics['cer']:.2%}, "
            f"WER={metrics['wer']:.2%}"
        )

    return phase1_metrics


def save_results(
    results_dir: Path,
    baseline_metrics: Dict[str, Dict[str, float]],
    phase1_metrics: Dict[str, Dict[str, float]],
    corrections: Dict[str, List[str]],
    data: Dict[str, List[Tuple[str, str]]],
    config: Dict[str, Any],
    num_samples: int = 20
) -> None:
    """
    Save all results to files.

    Args:
        results_dir: Directory to save results.
        baseline_metrics: Baseline evaluation metrics.
        phase1_metrics: Phase 1 evaluation metrics.
        corrections: Corrected texts by dataset.
        data: Original data pairs.
        config: Configuration used.
        num_samples: Number of sample corrections to save.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {results_dir}")

    # 1. Save metrics JSON
    all_metrics = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': config.get('model', {}).get('name', 'unknown'),
            'temperature': config.get('model', {}).get('temperature', 0.1),
        },
        'baseline': baseline_metrics,
        'phase1': phase1_metrics,
        'improvements': {}
    }

    # Calculate improvements
    for dataset_name in baseline_metrics:
        if dataset_name in phase1_metrics:
            improvement = calculate_improvement(
                baseline_metrics[dataset_name],
                phase1_metrics[dataset_name]
            )
            all_metrics['improvements'][dataset_name] = improvement

    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # 2. Save corrected texts
    for dataset_name, corrected_texts in corrections.items():
        if not corrected_texts:
            continue

        filename = f"{dataset_name.lower().replace('-', '_')}_corrected.txt"
        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            for text in corrected_texts:
                f.write(text + '\n---\n')

        logger.info(f"Saved {len(corrected_texts)} corrections to {filepath}")

    # 3. Save sample corrections
    samples_path = results_dir / 'sample_corrections.txt'
    with open(samples_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SAMPLE CORRECTIONS - Phase 1 LLM Baseline\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        sample_count = 0
        for dataset_name, pairs in data.items():
            corrected_list = corrections.get(dataset_name, [])
            if not pairs or not corrected_list:
                continue

            f.write(f"\n{'#' * 60}\n")
            f.write(f"# Dataset: {dataset_name}\n")
            f.write(f"{'#' * 60}\n")

            for i, ((ocr_text, ground_truth), corrected) in enumerate(
                zip(pairs[:num_samples], corrected_list[:num_samples])
            ):
                sample_count += 1
                sample_str = format_sample_correction(
                    sample_count, ocr_text, corrected, ground_truth
                )
                f.write(sample_str)

                if sample_count >= num_samples:
                    break

            if sample_count >= num_samples:
                break

    logger.info(f"Saved {sample_count} sample corrections to {samples_path}")

    # 4. Generate markdown report
    generate_report(
        results_dir / 'report.md',
        baseline_metrics,
        phase1_metrics,
        all_metrics['improvements'],
        config
    )


def generate_report(
    report_path: Path,
    baseline_metrics: Dict[str, Dict[str, float]],
    phase1_metrics: Dict[str, Dict[str, float]],
    improvements: Dict[str, Dict[str, float]],
    config: Dict[str, Any]
) -> None:
    """
    Generate markdown report comparing baseline vs Phase 1.

    Args:
        report_path: Path to save the report.
        baseline_metrics: Baseline evaluation metrics.
        phase1_metrics: Phase 1 evaluation metrics.
        improvements: Improvement percentages.
        config: Configuration used.
    """
    model_name = config.get('model', {}).get('name', 'Unknown')

    report = f"""# Phase 1 Report: LLM Baseline for Arabic Post-OCR Correction

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Model:** {model_name}
- **Temperature:** {config.get('model', {}).get('temperature', 0.1)}
- **Approach:** Zero-shot prompting

## Results Summary

### Baseline Metrics (OCR vs Ground Truth)

| Dataset | CER | WER | Samples |
|---------|-----|-----|---------|
"""

    for dataset_name, metrics in baseline_metrics.items():
        report += (
            f"| {dataset_name} | {metrics['cer']:.2%} | {metrics['wer']:.2%} | "
            f"{metrics.get('total_samples', 'N/A')} |\n"
        )

    report += """
### Phase 1 Metrics (LLM Corrected vs Ground Truth)

| Dataset | CER | WER | Samples |
|---------|-----|-----|---------|
"""

    for dataset_name, metrics in phase1_metrics.items():
        report += (
            f"| {dataset_name} | {metrics['cer']:.2%} | {metrics['wer']:.2%} | "
            f"{metrics.get('total_samples', 'N/A')} |\n"
        )

    report += """
### Improvement Analysis

| Dataset | CER Improvement | WER Improvement | CER Change | WER Change |
|---------|-----------------|-----------------|------------|------------|
"""

    for dataset_name, imp in improvements.items():
        cer_imp = imp.get('cer_improvement', 0)
        wer_imp = imp.get('wer_improvement', 0)
        cer_change = imp.get('cer_absolute_improvement', 0)
        wer_change = imp.get('wer_absolute_improvement', 0)

        # Format with direction indicator
        cer_dir = "+" if cer_imp > 0 else ""
        wer_dir = "+" if wer_imp > 0 else ""

        report += (
            f"| {dataset_name} | {cer_dir}{cer_imp:.1%} | {wer_dir}{wer_imp:.1%} | "
            f"{cer_change:+.2%} | {wer_change:+.2%} |\n"
        )

    report += """
## Detailed Metrics

"""

    for dataset_name in baseline_metrics:
        baseline = baseline_metrics[dataset_name]
        phase1 = phase1_metrics.get(dataset_name, {})
        imp = improvements.get(dataset_name, {})

        report += f"""### {dataset_name}

**Baseline (OCR):**
- CER: {baseline['cer']:.4f} ({baseline['cer']:.2%})
- WER: {baseline['wer']:.4f} ({baseline['wer']:.2%})
- Total characters: {baseline.get('total_ref_chars', 'N/A')}
- Total words: {baseline.get('total_ref_words', 'N/A')}

**Phase 1 (LLM Corrected):**
- CER: {phase1.get('cer', 0):.4f} ({phase1.get('cer', 0):.2%})
- WER: {phase1.get('wer', 0):.4f} ({phase1.get('wer', 0):.2%})

**Improvement:**
- CER reduced by: {imp.get('cer_absolute_improvement', 0):.4f} ({imp.get('cer_improvement', 0):.1%} relative)
- WER reduced by: {imp.get('wer_absolute_improvement', 0):.4f} ({imp.get('wer_improvement', 0):.1%} relative)

"""

    report += """## Conclusion

"""

    # Add summary conclusion
    total_cer_baseline = sum(m['cer'] for m in baseline_metrics.values()) / len(baseline_metrics) if baseline_metrics else 0
    total_cer_phase1 = sum(m['cer'] for m in phase1_metrics.values()) / len(phase1_metrics) if phase1_metrics else 0
    overall_improvement = (total_cer_baseline - total_cer_phase1) / total_cer_baseline if total_cer_baseline > 0 else 0

    if overall_improvement > 0:
        report += f"""The Phase 1 LLM baseline shows a **{overall_improvement:.1%} relative improvement** in CER across all datasets.
This demonstrates that zero-shot LLM correction can effectively reduce OCR errors in Arabic text.
"""
    else:
        report += """The Phase 1 LLM baseline shows mixed results. Further investigation and prompt optimization may be needed.
"""

    report += """
## Files Generated

- `metrics.json` - Complete metrics in JSON format
- `*_corrected.txt` - Corrected texts for each dataset
- `sample_corrections.txt` - Sample before/after corrections
- `report.md` - This report

## Next Steps

1. Analyze error patterns in sample corrections
2. Investigate cases where LLM correction degraded quality
3. Experiment with different prompts and models
4. Proceed to Phase 2 with specialized approaches
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Generated report at {report_path}")


def main():
    """Main entry point for Phase 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 1: LLM Baseline for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        help='Datasets to process (overrides config)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit samples per dataset (overrides config)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )
    parser.add_argument(
        '--skip-correction',
        action='store_true',
        help='Skip LLM correction (baseline only)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.datasets:
        config['data']['datasets'] = args.datasets
    if args.limit:
        config['processing']['limit_per_dataset'] = args.limit

    logger.info("=" * 60)
    logger.info("Phase 1: LLM Baseline for Arabic Post-OCR Correction")
    logger.info("=" * 60)

    # Initialize data loader
    try:
        data_loader = DataLoader(
            ground_truth_base=config['data']['ground_truth_base'],
            predictions_base=config['data']['predictions_base']
        )
    except FileNotFoundError as e:
        logger.error(f"Data directory not found: {e}")
        logger.info("Please ensure your data is organized as:")
        logger.info("  data/Original/<dataset>/*.txt  (ground truth)")
        logger.info("  data/Predictions/<dataset>/*.txt  (OCR predictions)")
        sys.exit(1)

    # Load datasets
    datasets = config['data']['datasets']
    limit = config['processing'].get('limit_per_dataset')

    logger.info(f"Loading datasets: {datasets}")
    if limit:
        logger.info(f"Limiting to {limit} samples per dataset")

    data = data_loader.load_multiple_datasets(
        datasets,
        limit_per_dataset=limit
    )

    total_samples = sum(len(pairs) for pairs in data.values())
    if total_samples == 0:
        logger.error("No data loaded. Please check your data directories.")
        sys.exit(1)

    logger.info(f"Loaded {total_samples} total samples")

    # Calculate baseline metrics
    baseline_metrics = run_baseline_evaluation(data)

    # Run LLM correction (unless skipped)
    corrections = {}
    phase1_metrics = {}

    if not args.skip_correction:
        try:
            logger.info(f"Initializing LLM: {config['model']['name']}")
            corrector = LLMCorrector(
                model_name=config['model']['name'],
                temperature=config['model'].get('temperature', 0.1),
                max_new_tokens=config['model'].get('max_new_tokens', 1024),
                load_in_4bit=config['model'].get('load_in_4bit', False),
                load_in_8bit=config['model'].get('load_in_8bit', False),
            )

            # Run corrections
            corrections = run_llm_correction(
                corrector, data, progress=not args.no_progress
            )

            # Calculate Phase 1 metrics
            phase1_metrics = run_phase1_evaluation(corrections, data)

        except LLMCorrectorError as e:
            logger.error(f"LLM initialization failed: {e}")
            logger.info("Saving baseline results only")
    else:
        logger.info("Skipping LLM correction (baseline only mode)")

    # Save results
    results_dir = Path(config['output']['results_dir'])
    num_samples = config['output'].get('num_samples_to_show', 20)

    save_results(
        results_dir,
        baseline_metrics,
        phase1_metrics,
        corrections,
        data,
        config,
        num_samples
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)

    print("\n" + format_metrics(
        {k: v for d in baseline_metrics.values() for k, v in d.items() if k in ['cer', 'wer']},
        "Average Baseline"
    ))

    if phase1_metrics:
        print("\n" + format_metrics(
            {k: v for d in phase1_metrics.values() for k, v in d.items() if k in ['cer', 'wer']},
            "Average Phase 1"
        ))

    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Report: {results_dir / 'report.md'}")


if __name__ == '__main__':
    main()
