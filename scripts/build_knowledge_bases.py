#!/usr/bin/env python3
"""
Master Knowledge Base Builder.

Orchestrates the construction of all knowledge bases for Arabic Post-OCR correction:
1. Qaari error confusion matrix
2. Arabic vocabulary from OpenITI
3. N-gram statistics
4. QALB error-correction pairs
5. Grammar/spelling rules

Usage:
    python scripts/build_knowledge_bases.py --config kb_config.yaml
    python scripts/build_knowledge_bases.py --all --openiti-path path/to/openiti

Example:
    # Build all knowledge bases
    python scripts/build_knowledge_bases.py --all

    # Build specific components
    python scripts/build_knowledge_bases.py --vocab --ngrams

Runtime Estimate:
    - Confusion matrix: 1-2 minutes
    - Vocabulary (100K): 5-15 minutes
    - N-grams: 10-20 minutes
    - QALB: 1-3 minutes
    - Rules: 5-30 minutes (with LLM)
    Total: ~30-60 minutes for full build
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('knowledge_base_build.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    """Configuration for knowledge base building."""
    # Data paths
    ground_truth_path: str = "data/Original"
    predictions_path: str = "data/Predictions"
    openiti_path: str = "data/openiti"
    qalb_path: str = "data/qalb"
    grammar_books_path: str = "data/grammar_books"

    # Output paths
    output_dir: str = "data"

    # Dataset names
    datasets: List[str] = None

    # Vocabulary settings
    vocab_top_n: int = 100000
    vocab_min_count: int = 2

    # N-gram settings
    ngrams_top_bigrams: int = 50000
    ngrams_top_trigrams: int = 50000
    ngrams_min_count: int = 2

    # QALB settings
    qalb_few_shot_n: int = 20
    qalb_context_window: int = 2

    # Rules settings
    rules_model: str = "Qwen/Qwen2.5-3B-Instruct"
    rules_default_only: bool = False

    # Processing
    skip_existing: bool = True
    force_rebuild: bool = False

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["PATS-A01", "KHATT"]


@dataclass
class BuildResult:
    """Result of a build step."""
    name: str
    success: bool
    output_files: List[str]
    duration_seconds: float
    error_message: str = ""
    stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.stats is None:
            self.stats = {}


class KnowledgeBaseBuilder:
    """
    Orchestrate knowledge base construction.

    Example:
        >>> builder = KnowledgeBaseBuilder(config)
        >>> results = builder.build_all()
        >>> builder.generate_report(results)
    """

    def __init__(self, config: BuildConfig):
        """
        Initialize the builder.

        Args:
            config: Build configuration.
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _file_exists(self, filename: str) -> bool:
        """Check if output file already exists."""
        filepath = self.output_dir / filename
        return filepath.exists() and filepath.stat().st_size > 0

    def _should_build(self, filename: str) -> bool:
        """Determine if we should build this component."""
        if self.config.force_rebuild:
            return True
        if self.config.skip_existing and self._file_exists(filename):
            logger.info(f"Skipping {filename} (already exists)")
            return False
        return True

    def build_confusion_matrix(self) -> BuildResult:
        """Build Qaari error confusion matrix."""
        start_time = time.time()
        output_files = ['confusion_matrix.json', 'error_statistics.json']

        if not self._should_build('confusion_matrix.json'):
            return BuildResult(
                name="confusion_matrix",
                success=True,
                output_files=output_files,
                duration_seconds=0,
                stats={'skipped': True}
            )

        logger.info("Building confusion matrix...")

        try:
            from src.data_loader import DataLoader
            from scripts.analyze_qaari_errors import analyze_qaari_errors, save_results

            data_loader = DataLoader(
                ground_truth_base=self.config.ground_truth_path,
                predictions_base=self.config.predictions_path
            )

            confusion_matrix, statistics = analyze_qaari_errors(
                self.config.datasets,
                data_loader
            )

            save_results(confusion_matrix, statistics, self.output_dir)

            duration = time.time() - start_time
            return BuildResult(
                name="confusion_matrix",
                success=True,
                output_files=output_files,
                duration_seconds=duration,
                stats={
                    'total_chars': statistics.total_chars,
                    'total_errors': statistics.total_errors,
                    'error_rate': statistics.total_errors / statistics.total_chars if statistics.total_chars > 0 else 0
                }
            )

        except Exception as e:
            logger.error(f"Failed to build confusion matrix: {e}")
            return BuildResult(
                name="confusion_matrix",
                success=False,
                output_files=[],
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def build_vocabulary(self) -> BuildResult:
        """Build vocabulary from OpenITI corpus."""
        start_time = time.time()
        output_file = 'vocab_100k.json'

        if not self._should_build(output_file):
            return BuildResult(
                name="vocabulary",
                success=True,
                output_files=[output_file],
                duration_seconds=0,
                stats={'skipped': True}
            )

        logger.info("Building vocabulary...")

        try:
            from scripts.build_vocabulary import build_vocabulary_from_openiti, save_vocabulary

            vocabulary, metadata = build_vocabulary_from_openiti(
                self.config.openiti_path,
                top_n=self.config.vocab_top_n,
                min_count=self.config.vocab_min_count
            )

            output_path = self.output_dir / output_file
            save_vocabulary(vocabulary, metadata, output_path)

            duration = time.time() - start_time
            return BuildResult(
                name="vocabulary",
                success=True,
                output_files=[output_file],
                duration_seconds=duration,
                stats={
                    'vocabulary_size': len(vocabulary),
                    'total_words_processed': metadata.get('total_words', 0)
                }
            )

        except Exception as e:
            logger.error(f"Failed to build vocabulary: {e}")
            return BuildResult(
                name="vocabulary",
                success=False,
                output_files=[],
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def build_ngrams(self) -> BuildResult:
        """Build n-gram statistics from OpenITI corpus."""
        start_time = time.time()
        output_file = 'ngrams.json'

        if not self._should_build(output_file):
            return BuildResult(
                name="ngrams",
                success=True,
                output_files=[output_file],
                duration_seconds=0,
                stats={'skipped': True}
            )

        logger.info("Building n-gram statistics...")

        try:
            from scripts.build_ngrams import build_ngram_statistics, save_ngrams

            bigrams, trigrams, metadata = build_ngram_statistics(
                self.config.openiti_path,
                top_bigrams=self.config.ngrams_top_bigrams,
                top_trigrams=self.config.ngrams_top_trigrams,
                min_count=self.config.ngrams_min_count
            )

            output_path = self.output_dir / output_file
            save_ngrams(bigrams, trigrams, metadata, output_path)

            duration = time.time() - start_time
            return BuildResult(
                name="ngrams",
                success=True,
                output_files=[output_file],
                duration_seconds=duration,
                stats={
                    'num_bigrams': len(bigrams),
                    'num_trigrams': len(trigrams)
                }
            )

        except Exception as e:
            logger.error(f"Failed to build n-grams: {e}")
            return BuildResult(
                name="ngrams",
                success=False,
                output_files=[],
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def build_qalb(self) -> BuildResult:
        """Extract QALB error-correction pairs."""
        start_time = time.time()
        output_files = ['qalb_corrections.json', 'qalb_few_shot.json']

        if not self._should_build('qalb_corrections.json'):
            return BuildResult(
                name="qalb",
                success=True,
                output_files=output_files,
                duration_seconds=0,
                stats={'skipped': True}
            )

        logger.info("Extracting QALB corrections...")

        try:
            from scripts.extract_qalb import (
                extract_qalb_corrections,
                select_few_shot_examples,
                save_corrections,
                save_few_shot
            )

            corrections, stats = extract_qalb_corrections(
                self.config.qalb_path,
                context_window=self.config.qalb_context_window
            )

            if not corrections:
                raise ValueError("No corrections extracted from QALB")

            save_corrections(
                corrections, stats,
                self.output_dir / 'qalb_corrections.json'
            )

            few_shot = select_few_shot_examples(
                corrections,
                n=self.config.qalb_few_shot_n
            )
            save_few_shot(few_shot, self.output_dir / 'qalb_few_shot.json')

            duration = time.time() - start_time
            return BuildResult(
                name="qalb",
                success=True,
                output_files=output_files,
                duration_seconds=duration,
                stats={
                    'total_corrections': len(corrections),
                    'few_shot_examples': len(few_shot)
                }
            )

        except Exception as e:
            logger.error(f"Failed to build QALB: {e}")
            return BuildResult(
                name="qalb",
                success=False,
                output_files=[],
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def build_rules(self) -> BuildResult:
        """Extract grammar/spelling rules."""
        start_time = time.time()
        output_files = [
            'rules/morphology_rules.json',
            'rules/syntax_rules.json',
            'rules/orthography_rules.json'
        ]

        if not self._should_build('rules/orthography_rules.json'):
            return BuildResult(
                name="rules",
                success=True,
                output_files=output_files,
                duration_seconds=0,
                stats={'skipped': True}
            )

        logger.info("Extracting grammar rules...")

        try:
            from scripts.extract_rules import (
                RuleExtractor,
                extract_rules_from_books,
                create_default_rules,
                deduplicate_rules,
                save_rules_by_category
            )

            rules_dir = self.output_dir / 'rules'

            if self.config.rules_default_only:
                rules = create_default_rules()
            else:
                # Check if grammar books exist
                books_path = Path(self.config.grammar_books_path)
                if books_path.exists() and any(books_path.glob('**/*.txt')):
                    extractor = RuleExtractor(model_name=self.config.rules_model)
                    rules, _ = extract_rules_from_books(
                        [str(books_path)],
                        extractor
                    )
                    # Add default rules
                    rules.extend(create_default_rules())
                else:
                    logger.warning("No grammar books found, using default rules only")
                    rules = create_default_rules()

            rules = deduplicate_rules(rules)
            save_rules_by_category(rules, rules_dir)

            duration = time.time() - start_time
            return BuildResult(
                name="rules",
                success=True,
                output_files=output_files,
                duration_seconds=duration,
                stats={
                    'total_rules': len(rules),
                    'morphology': len([r for r in rules if r.category == 'morphology']),
                    'syntax': len([r for r in rules if r.category == 'syntax']),
                    'orthography': len([r for r in rules if r.category == 'orthography'])
                }
            )

        except Exception as e:
            logger.error(f"Failed to build rules: {e}")
            return BuildResult(
                name="rules",
                success=False,
                output_files=[],
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )

    def build_all(self) -> List[BuildResult]:
        """
        Build all knowledge bases.

        Returns:
            List of BuildResult for each component.
        """
        results = []

        # Build in order of dependency
        logger.info("=" * 60)
        logger.info("KNOWLEDGE BASE BUILD - STARTING")
        logger.info("=" * 60)

        # 1. Confusion matrix (requires OCR data)
        result = self.build_confusion_matrix()
        results.append(result)
        self._log_result(result)

        # 2. Vocabulary (requires OpenITI)
        result = self.build_vocabulary()
        results.append(result)
        self._log_result(result)

        # 3. N-grams (requires OpenITI)
        result = self.build_ngrams()
        results.append(result)
        self._log_result(result)

        # 4. QALB (requires QALB corpus)
        result = self.build_qalb()
        results.append(result)
        self._log_result(result)

        # 5. Rules (optionally requires grammar books)
        result = self.build_rules()
        results.append(result)
        self._log_result(result)

        return results

    def _log_result(self, result: BuildResult) -> None:
        """Log a build result."""
        if result.success:
            if result.stats.get('skipped'):
                logger.info(f"  {result.name}: SKIPPED (already exists)")
            else:
                logger.info(f"  {result.name}: SUCCESS ({result.duration_seconds:.1f}s)")
        else:
            logger.error(f"  {result.name}: FAILED - {result.error_message}")

    def generate_report(self, results: List[BuildResult]) -> str:
        """
        Generate a summary report of the build.

        Args:
            results: List of build results.

        Returns:
            Report string.
        """
        report = []
        report.append("=" * 60)
        report.append("KNOWLEDGE BASE BUILD REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")

        total_time = sum(r.duration_seconds for r in results)
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        report.append(f"Total time: {total_time:.1f} seconds")
        report.append(f"Successful: {successful}/{len(results)}")
        report.append(f"Failed: {failed}/{len(results)}")
        report.append("")

        report.append("Component Results:")
        report.append("-" * 40)

        for result in results:
            status = "SUCCESS" if result.success else "FAILED"
            if result.stats.get('skipped'):
                status = "SKIPPED"

            report.append(f"\n{result.name}:")
            report.append(f"  Status: {status}")
            report.append(f"  Duration: {result.duration_seconds:.1f}s")

            if result.output_files:
                report.append(f"  Output files:")
                for f in result.output_files:
                    filepath = self.output_dir / f
                    if filepath.exists():
                        size = filepath.stat().st_size / 1024
                        report.append(f"    - {f} ({size:.1f} KB)")
                    else:
                        report.append(f"    - {f} (not created)")

            if result.stats and not result.stats.get('skipped'):
                report.append(f"  Statistics:")
                for key, value in result.stats.items():
                    if key != 'skipped':
                        if isinstance(value, float):
                            report.append(f"    - {key}: {value:.4f}")
                        else:
                            report.append(f"    - {key}: {value}")

            if result.error_message:
                report.append(f"  Error: {result.error_message}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def validate_outputs(self) -> Dict[str, bool]:
        """
        Validate that all expected output files exist and are valid.

        Returns:
            Dictionary mapping filename to validity status.
        """
        expected_files = [
            'confusion_matrix.json',
            'error_statistics.json',
            'vocab_100k.json',
            'ngrams.json',
            'qalb_corrections.json',
            'qalb_few_shot.json',
            'rules/morphology_rules.json',
            'rules/syntax_rules.json',
            'rules/orthography_rules.json',
        ]

        validity = {}

        for filename in expected_files:
            filepath = self.output_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    validity[filename] = bool(data)
                except Exception:
                    validity[filename] = False
            else:
                validity[filename] = False

        return validity


def load_config(config_path: str) -> BuildConfig:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return BuildConfig()

    with open(config_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return BuildConfig(**data)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build all knowledge bases for Arabic Post-OCR correction"
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Build all knowledge bases'
    )
    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Build confusion matrix only'
    )
    parser.add_argument(
        '--vocab',
        action='store_true',
        help='Build vocabulary only'
    )
    parser.add_argument(
        '--ngrams',
        action='store_true',
        help='Build n-grams only'
    )
    parser.add_argument(
        '--qalb',
        action='store_true',
        help='Build QALB corrections only'
    )
    parser.add_argument(
        '--rules',
        action='store_true',
        help='Build grammar rules only'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force rebuild even if files exist'
    )
    parser.add_argument(
        '--openiti-path',
        help='Path to OpenITI corpus'
    )
    parser.add_argument(
        '--qalb-path',
        help='Path to QALB corpus'
    )
    parser.add_argument(
        '--output', '-o',
        default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--default-rules-only',
        action='store_true',
        help='Only use default rules (no LLM extraction)'
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = BuildConfig()

    # Override with command line arguments
    if args.force:
        config.force_rebuild = True
    if args.openiti_path:
        config.openiti_path = args.openiti_path
    if args.qalb_path:
        config.qalb_path = args.qalb_path
    if args.output:
        config.output_dir = args.output
    if args.default_rules_only:
        config.rules_default_only = True

    # Create builder
    builder = KnowledgeBaseBuilder(config)

    # Determine what to build
    build_specific = any([
        args.confusion_matrix, args.vocab, args.ngrams,
        args.qalb, args.rules
    ])

    results = []

    if args.all or not build_specific:
        results = builder.build_all()
    else:
        if args.confusion_matrix:
            results.append(builder.build_confusion_matrix())
        if args.vocab:
            results.append(builder.build_vocabulary())
        if args.ngrams:
            results.append(builder.build_ngrams())
        if args.qalb:
            results.append(builder.build_qalb())
        if args.rules:
            results.append(builder.build_rules())

    # Generate and print report
    report = builder.generate_report(results)
    print(report)

    # Validate outputs
    validity = builder.validate_outputs()
    print("\nOutput Validation:")
    for filename, valid in validity.items():
        status = "VALID" if valid else "MISSING/INVALID"
        print(f"  {filename}: {status}")

    # Save report
    report_path = Path(config.output_dir) / 'build_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
