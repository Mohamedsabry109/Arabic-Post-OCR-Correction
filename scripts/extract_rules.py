#!/usr/bin/env python3
"""
Arabic Grammar Rule Extractor using LLM.

Extracts structured grammar rules from Arabic grammar/spelling books
using an LLM to parse and categorize the rules.

Usage:
    python scripts/extract_rules.py --books path/to/books --model Qwen/Qwen2.5-3B-Instruct
    python scripts/extract_rules.py --books data/grammar_books --output data/rules

Example Output:
    data/rules/
    ├── morphology_rules.json
    ├── syntax_rules.json
    └── orthography_rules.json

Runtime Estimate:
    ~5-30 minutes depending on number of books and model speed
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime

from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available, LLM extraction disabled")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Rule categories
RULE_CATEGORIES = ['morphology', 'syntax', 'orthography']


# LLM prompt template for rule extraction
RULE_EXTRACTION_PROMPT = """أنت خبير في قواعد اللغة العربية. استخرج القواعد الإملائية والنحوية والصرفية من النص التالي.

لكل قاعدة، قدم:
1. اسم القاعدة (rule_name)
2. وصف مختصر (description)
3. التصنيف: morphology (صرف) أو syntax (نحو) أو orthography (إملاء)
4. أمثلة صحيحة (correct_examples)
5. أمثلة خاطئة شائعة (incorrect_examples)
6. متى تُطبق القاعدة (when_to_apply)

النص:
{text}

أجب بصيغة JSON فقط بالشكل التالي:
{{
  "rules": [
    {{
      "rule_name": "اسم القاعدة",
      "description": "وصف القاعدة",
      "category": "orthography",
      "correct_examples": ["مثال صحيح 1", "مثال صحيح 2"],
      "incorrect_examples": ["مثال خاطئ 1", "مثال خاطئ 2"],
      "when_to_apply": "متى تطبق القاعدة"
    }}
  ]
}}"""

RULE_EXTRACTION_PROMPT_EN = """You are an expert in Arabic grammar. Extract spelling, grammatical, and morphological rules from the following text.

For each rule, provide:
1. Rule name (rule_name) - in Arabic
2. Brief description (description) - in Arabic
3. Category: morphology, syntax, or orthography
4. Correct examples (correct_examples) - in Arabic
5. Common incorrect examples (incorrect_examples) - in Arabic
6. When to apply (when_to_apply) - in Arabic

Text:
{text}

Respond ONLY with valid JSON in this format:
{{
  "rules": [
    {{
      "rule_name": "اسم القاعدة",
      "description": "وصف القاعدة",
      "category": "orthography",
      "correct_examples": ["مثال صحيح 1"],
      "incorrect_examples": ["مثال خاطئ 1"],
      "when_to_apply": "متى تطبق القاعدة"
    }}
  ]
}}"""


@dataclass
class Rule:
    """A single grammar/spelling rule."""
    rule_name: str
    description: str
    category: str
    correct_examples: List[str]
    incorrect_examples: List[str]
    when_to_apply: str
    source_file: str = ""
    rule_hash: str = ""

    def __post_init__(self):
        # Generate unique hash for deduplication
        content = f"{self.rule_name}|{self.description}|{self.category}"
        self.rule_hash = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class ExtractionStatistics:
    """Statistics about rule extraction."""
    total_books: int = 0
    total_chunks: int = 0
    total_rules: int = 0
    rules_by_category: Dict[str, int] = None
    failed_extractions: int = 0

    def __post_init__(self):
        if self.rules_by_category is None:
            self.rules_by_category = defaultdict(int)


class RuleExtractor:
    """
    Extract grammar rules from books using LLM.

    Example:
        >>> extractor = RuleExtractor(model_name="Qwen/Qwen2.5-3B-Instruct")
        >>> rules = extractor.extract_from_text("نص يحتوي على قواعد...")
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        max_chunk_size: int = 2000,
        temperature: float = 0.1
    ):
        """
        Initialize the rule extractor.

        Args:
            model_name: HuggingFace model name.
            device: Device to use ("auto", "cuda", "cpu").
            max_chunk_size: Maximum characters per chunk.
            temperature: Generation temperature.
        """
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self._device = None

        if HAS_TRANSFORMERS:
            self._load_model(device)
        else:
            logger.warning("Running without LLM - will use rule templates only")

    def _load_model(self, device: str) -> None:
        """Load the LLM model."""
        logger.info(f"Loading model: {self.model_name}")

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if self._device == "cuda" else None
            )

            if self._device != "cuda":
                self.model = self.model.to(self._device)

            self.model.eval()
            logger.info(f"Model loaded on {self._device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into processable chunks.

        Tries to split at paragraph boundaries.
        """
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            cleaned = re.sub(r',\s*}', '}', response)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return None

    def extract_from_text(self, text: str, source_file: str = "") -> List[Rule]:
        """
        Extract rules from a text passage.

        Args:
            text: Text containing grammar rules.
            source_file: Source file name for attribution.

        Returns:
            List of extracted Rule objects.
        """
        if not self.model:
            logger.warning("No model loaded, returning empty rules")
            return []

        rules = []

        # Chunk the text
        chunks = self._chunk_text(text)

        for chunk in chunks:
            if len(chunk) < 50:  # Skip very short chunks
                continue

            try:
                # Build prompt
                prompt = RULE_EXTRACTION_PROMPT_EN.format(text=chunk)

                # Generate
                messages = [
                    {"role": "system", "content": "You extract Arabic grammar rules and respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ]

                if hasattr(self.tokenizer, 'apply_chat_template'):
                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    input_text = prompt

                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self._device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Parse response
                result = self._extract_json_from_response(response)

                if result and 'rules' in result:
                    for rule_data in result['rules']:
                        try:
                            rule = Rule(
                                rule_name=rule_data.get('rule_name', ''),
                                description=rule_data.get('description', ''),
                                category=rule_data.get('category', 'orthography'),
                                correct_examples=rule_data.get('correct_examples', []),
                                incorrect_examples=rule_data.get('incorrect_examples', []),
                                when_to_apply=rule_data.get('when_to_apply', ''),
                                source_file=source_file
                            )

                            # Validate category
                            if rule.category not in RULE_CATEGORIES:
                                rule.category = 'orthography'

                            rules.append(rule)

                        except Exception as e:
                            logger.debug(f"Error parsing rule: {e}")

            except Exception as e:
                logger.warning(f"Error extracting from chunk: {e}")

        return rules


def extract_rules_from_books(
    book_paths: List[str],
    extractor: RuleExtractor,
    show_progress: bool = True
) -> Tuple[List[Rule], ExtractionStatistics]:
    """
    Extract rules from multiple grammar books.

    Args:
        book_paths: List of paths to book text files.
        extractor: RuleExtractor instance.
        show_progress: Show progress bar.

    Returns:
        Tuple of (list of Rules, statistics).

    Example:
        >>> extractor = RuleExtractor()
        >>> rules, stats = extract_rules_from_books(["book1.txt", "book2.txt"], extractor)
    """
    all_rules = []
    stats = ExtractionStatistics()

    book_files = []
    for path in book_paths:
        path = Path(path)
        if path.is_file():
            book_files.append(path)
        elif path.is_dir():
            book_files.extend(path.glob('**/*.txt'))

    stats.total_books = len(book_files)
    logger.info(f"Processing {stats.total_books} book files")

    iterator = tqdm(book_files, desc="Extracting rules") if show_progress else book_files

    for book_file in iterator:
        try:
            # Read book content
            encodings = ['utf-8', 'utf-8-sig', 'cp1256']
            content = None

            for encoding in encodings:
                try:
                    with open(book_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if not content:
                logger.warning(f"Could not read: {book_file}")
                stats.failed_extractions += 1
                continue

            # Extract rules
            rules = extractor.extract_from_text(content, str(book_file))
            all_rules.extend(rules)

            for rule in rules:
                stats.rules_by_category[rule.category] += 1

            stats.total_rules += len(rules)
            stats.total_chunks += len(extractor._chunk_text(content))

        except Exception as e:
            logger.warning(f"Error processing {book_file}: {e}")
            stats.failed_extractions += 1

    return all_rules, stats


def deduplicate_rules(rules: List[Rule]) -> List[Rule]:
    """
    Remove duplicate rules based on content hash.

    Args:
        rules: List of rules with potential duplicates.

    Returns:
        Deduplicated list of rules.
    """
    seen_hashes = set()
    unique_rules = []

    for rule in rules:
        if rule.rule_hash not in seen_hashes:
            seen_hashes.add(rule.rule_hash)
            unique_rules.append(rule)

    logger.info(f"Deduplicated {len(rules)} -> {len(unique_rules)} rules")
    return unique_rules


def save_rules_by_category(
    rules: List[Rule],
    output_dir: Path
) -> None:
    """
    Save rules to separate files by category.

    Args:
        rules: List of rules to save.
        output_dir: Directory to save rule files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by category
    by_category: Dict[str, List[Rule]] = defaultdict(list)
    for rule in rules:
        by_category[rule.category].append(rule)

    # Save each category
    for category in RULE_CATEGORIES:
        category_rules = by_category.get(category, [])

        output = {
            'metadata': {
                'category': category,
                'num_rules': len(category_rules),
                'created_at': datetime.now().isoformat(),
            },
            'rules': [asdict(r) for r in category_rules]
        }

        output_path = output_dir / f'{category}_rules.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(category_rules)} {category} rules to {output_path}")


def create_default_rules() -> List[Rule]:
    """
    Create default Arabic spelling/grammar rules.

    Returns a set of common rules when LLM extraction is not available.
    """
    default_rules = [
        Rule(
            rule_name="همزة الوصل والقطع",
            description="التمييز بين همزة الوصل (ا) وهمزة القطع (أ/إ)",
            category="orthography",
            correct_examples=["أَكَلَ", "استَغفَرَ", "إنّ"],
            incorrect_examples=["اكل", "إستغفر", "ان"],
            when_to_apply="في بداية الكلمات العربية"
        ),
        Rule(
            rule_name="التاء المربوطة والتاء المفتوحة",
            description="التمييز بين التاء المربوطة (ة) والتاء المفتوحة (ت)",
            category="orthography",
            correct_examples=["مدرسة", "كتبت", "فاطمة"],
            incorrect_examples=["مدرست", "كتبة", "فاطمت"],
            when_to_apply="في نهاية الأسماء والأفعال"
        ),
        Rule(
            rule_name="الألف المقصورة والياء",
            description="التمييز بين الألف المقصورة (ى) والياء (ي)",
            category="orthography",
            correct_examples=["على", "مصطفى", "هدي"],
            incorrect_examples=["علي", "مصطفي", "هدى"],
            when_to_apply="في نهاية الكلمات"
        ),
        Rule(
            rule_name="لام الشمسية والقمرية",
            description="قواعد إدغام اللام مع الحروف الشمسية والقمرية",
            category="morphology",
            correct_examples=["الشمس", "القمر", "النور"],
            incorrect_examples=["اشمس", "الكمر"],
            when_to_apply="عند استخدام أل التعريف"
        ),
        Rule(
            rule_name="التنوين",
            description="قواعد كتابة التنوين على الحروف",
            category="orthography",
            correct_examples=["كتاباً", "بيتٌ", "قلمٍ"],
            incorrect_examples=["كتابن", "بيتن", "قلمن"],
            when_to_apply="على الأسماء المنونة"
        ),
        Rule(
            rule_name="الألف الفارقة",
            description="إضافة الألف بعد واو الجماعة في الأفعال",
            category="orthography",
            correct_examples=["كتبوا", "ذهبوا", "لم يكتبوا"],
            incorrect_examples=["كتبو", "ذهبو", "لم يكتبو"],
            when_to_apply="بعد واو الجماعة في الأفعال"
        ),
        Rule(
            rule_name="همزة المد",
            description="كتابة همزة المد (آ) بدلاً من همزتين",
            category="orthography",
            correct_examples=["آمن", "القرآن", "آخر"],
            incorrect_examples=["أأمن", "القرأان", "أاخر"],
            when_to_apply="عند توالي همزتين أولاهما ساكنة"
        ),
    ]

    for rule in default_rules:
        rule.source_file = "default_rules"

    return default_rules


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Arabic grammar rules from books using LLM"
    )
    parser.add_argument(
        '--books', '-b',
        nargs='+',
        help='Paths to book files or directories'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/rules',
        help='Output directory for rule files'
    )
    parser.add_argument(
        '--model', '-m',
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Model name for extraction'
    )
    parser.add_argument(
        '--default-only',
        action='store_true',
        help='Only generate default rules (no LLM extraction)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Arabic Grammar Rule Extractor")
    logger.info("=" * 60)

    output_dir = Path(args.output)

    if args.default_only or not args.books:
        # Generate default rules only
        logger.info("Generating default rules...")
        rules = create_default_rules()
        stats = ExtractionStatistics(
            total_rules=len(rules),
            rules_by_category={r.category: 1 for r in rules}
        )
    else:
        # Extract from books using LLM
        if not HAS_TRANSFORMERS:
            logger.error("transformers library required for LLM extraction")
            logger.info("Use --default-only to generate default rules")
            sys.exit(1)

        extractor = RuleExtractor(model_name=args.model)
        rules, stats = extract_rules_from_books(
            args.books,
            extractor,
            show_progress=not args.no_progress
        )

        # Add default rules
        default_rules = create_default_rules()
        rules.extend(default_rules)

    # Deduplicate
    rules = deduplicate_rules(rules)

    # Save by category
    save_rules_by_category(rules, output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RULE EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rules extracted: {len(rules)}")

    for category in RULE_CATEGORIES:
        count = len([r for r in rules if r.category == category])
        logger.info(f"  {category}: {count}")

    logger.info(f"\nRules saved to: {output_dir}")


if __name__ == '__main__':
    main()
