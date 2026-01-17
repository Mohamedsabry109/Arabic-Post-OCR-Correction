"""
LLM-based Arabic OCR text corrector.

This module provides an LLM-based approach to correct OCR errors in Arabic text
using models like Qwen3-4B-Instruct or Qwen2.5-3B-Instruct with zero-shot prompting.

Example:
    >>> from src.llm_corrector import LLMCorrector
    >>> corrector = LLMCorrector(model_name="Qwen/Qwen2.5-3B-Instruct")
    >>> corrected = corrector.correct("مرحيا بالعالم")
    >>> print(corrected)
    مرحباً بالعالم
"""

import logging
import re
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import clean_text


# Configure logging
logger = logging.getLogger(__name__)


# Default Arabic system prompt for OCR correction
DEFAULT_SYSTEM_PROMPT_AR = """أنت مصحح نصوص عربية متخصص في تصحيح أخطاء التعرف الضوئي على الحروف (OCR).

مهمتك:
- صحح أخطاء OCR في النص المدخل
- حافظ على المعنى الأصلي للنص
- لا تضف أو تحذف محتوى
- أعد النص المصحح فقط بدون أي شرح

قواعد مهمة:
- صحح الأخطاء الإملائية الناتجة عن OCR
- صحح الحروف المشوهة أو المستبدلة
- حافظ على علامات الترقيم والتشكيل إن وجدت
- لا تغير أسلوب الكتابة أو تعيد صياغة الجمل"""

DEFAULT_SYSTEM_PROMPT_EN = """You are an Arabic text correction specialist for OCR errors.

Your task:
- Correct OCR errors in the input text
- Preserve the original meaning
- Do not add or remove content
- Return only the corrected text without any explanation

Important rules:
- Fix spelling errors caused by OCR
- Fix distorted or substituted characters
- Preserve punctuation and diacritics if present
- Do not change writing style or rephrase sentences"""


@dataclass
class CorrectorConfig:
    """Configuration for the LLM corrector."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 0.1
    max_new_tokens: int = 1024
    top_p: float = 0.9
    do_sample: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_AR
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "auto"


class LLMCorrectorError(Exception):
    """Base exception for LLM corrector errors."""
    pass


class ModelLoadError(LLMCorrectorError):
    """Raised when model fails to load."""
    pass


class CorrectionError(LLMCorrectorError):
    """Raised when correction fails."""
    pass


class LLMCorrector:
    """
    LLM-based Arabic OCR text corrector.

    Uses transformer models for zero-shot OCR error correction in Arabic text.
    Supports various Qwen models and other instruction-tuned LLMs.

    Attributes:
        model_name: Name or path of the model.
        config: CorrectorConfig instance with generation parameters.
        model: Loaded transformer model.
        tokenizer: Model tokenizer.

    Example:
        >>> corrector = LLMCorrector(
        ...     model_name="Qwen/Qwen2.5-3B-Instruct",
        ...     temperature=0.1
        ... )
        >>> result = corrector.correct("مرحيا بالعالم")
        >>> print(result)
        مرحباً بالعالم

        >>> # Batch correction
        >>> texts = ["نص اول", "نص ثاني"]
        >>> results = corrector.correct_batch(texts)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM corrector.

        Args:
            model_name: HuggingFace model name or local path.
                Recommended: "Qwen/Qwen2.5-3B-Instruct" or "Qwen/Qwen3-4B-Instruct-2507"
            temperature: Sampling temperature (lower = more deterministic).
                Defaults to 0.1 for consistent corrections.
            max_new_tokens: Maximum tokens to generate. Defaults to 1024.
            system_prompt: Custom system prompt. Defaults to Arabic prompt.
            device: Device to use ("auto", "cuda", "cpu"). Defaults to "auto".
            load_in_8bit: Load model in 8-bit quantization.
            load_in_4bit: Load model in 4-bit quantization.
            torch_dtype: Torch dtype for model ("float16", "bfloat16", "auto").
            **kwargs: Additional generation parameters.

        Raises:
            ModelLoadError: If model fails to load.
        """
        self.config = CorrectorConfig(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT_AR,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype or "auto",
        )

        # Store additional kwargs
        self.generation_kwargs = kwargs

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._device = None

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the model and tokenizer.

        Raises:
            ModelLoadError: If model fails to load.
        """
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device

            logger.info(f"Using device: {self._device}")

            # Configure quantization if requested
            quantization_config = None
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Determine torch dtype
            if self.config.torch_dtype == "auto":
                torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
            elif self.config.torch_dtype == "float16":
                torch_dtype = torch.float16
            elif self.config.torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            elif self._device == "cuda":
                model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )

            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self._device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def _build_prompt(self, ocr_text: str) -> str:
        """
        Build the prompt for correction.

        Args:
            ocr_text: OCR text to correct.

        Returns:
            Formatted prompt string.
        """
        # Use chat template if available
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"صحح النص التالي:\n{ocr_text}"}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            prompt = (
                f"<|system|>\n{self.config.system_prompt}<|end|>\n"
                f"<|user|>\nصحح النص التالي:\n{ocr_text}<|end|>\n"
                f"<|assistant|>\n"
            )

        return prompt

    def correct(
        self,
        ocr_text: str,
        clean_output: bool = True
    ) -> str:
        """
        Correct OCR errors in Arabic text.

        Args:
            ocr_text: Text with OCR errors to correct.
            clean_output: If True, cleans whitespace in output.

        Returns:
            Corrected text.

        Raises:
            CorrectionError: If correction fails.

        Example:
            >>> corrector = LLMCorrector()
            >>> corrected = corrector.correct("مرحيا بالعالم")
            >>> print(corrected)
            مرحباً بالعالم
        """
        if not ocr_text or not ocr_text.strip():
            return ""

        try:
            # Build prompt
            prompt = self._build_prompt(ocr_text)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self._device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **self.generation_kwargs
                )

            # Decode output
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            corrected_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            # Clean up the output
            corrected_text = self._clean_response(corrected_text)

            if clean_output:
                corrected_text = clean_text(corrected_text)

            return corrected_text

        except Exception as e:
            logger.error(f"Correction failed: {e}")
            raise CorrectionError(f"Failed to correct text: {e}")

    def _clean_response(self, response: str) -> str:
        """
        Clean the model's response to extract only the corrected text.

        Args:
            response: Raw model response.

        Returns:
            Cleaned corrected text.
        """
        # Remove common prefixes/suffixes that models might add
        patterns_to_remove = [
            r'^النص المصحح[:\s]*',
            r'^التصحيح[:\s]*',
            r'^الإجابة[:\s]*',
            r'^النص بعد التصحيح[:\s]*',
            r'^\s*[-•]\s*',
        ]

        cleaned = response.strip()
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)

        # Remove any trailing explanation
        # Look for common explanation patterns and cut there
        explanation_markers = [
            '\n\nملاحظة',
            '\n\nتم تصحيح',
            '\n\nالتصحيحات',
            '\nNote:',
            '\nExplanation:',
        ]
        for marker in explanation_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]

        return cleaned.strip()

    def correct_batch(
        self,
        texts: List[str],
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Correct multiple texts.

        Args:
            texts: List of texts to correct.
            batch_size: Number of texts to process at once.
                Currently only batch_size=1 is supported for quality.
            progress_callback: Optional callback function(current, total)
                for progress updates.

        Returns:
            List of corrected texts.

        Example:
            >>> corrector = LLMCorrector()
            >>> texts = ["نص اول", "نص ثاني"]
            >>> corrected = corrector.correct_batch(texts)
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            try:
                corrected = self.correct(text)
                results.append(corrected)
            except CorrectionError as e:
                logger.warning(f"Failed to correct text {i+1}: {e}")
                # Return original text on failure
                results.append(text)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        info = {
            'model_name': self.config.model_name,
            'device': self._device,
            'temperature': self.config.temperature,
            'max_new_tokens': self.config.max_new_tokens,
            'quantization': 'none',
        }

        if self.config.load_in_4bit:
            info['quantization'] = '4-bit'
        elif self.config.load_in_8bit:
            info['quantization'] = '8-bit'

        if self.model is not None:
            info['num_parameters'] = sum(
                p.numel() for p in self.model.parameters()
            )
            info['model_dtype'] = str(next(self.model.parameters()).dtype)

        return info

    def __repr__(self) -> str:
        return (
            f"LLMCorrector(model={self.config.model_name}, "
            f"device={self._device}, temp={self.config.temperature})"
        )


def create_corrector_from_config(config: Dict[str, Any]) -> LLMCorrector:
    """
    Create an LLMCorrector from a configuration dictionary.

    Args:
        config: Configuration dictionary with corrector settings.

    Returns:
        Configured LLMCorrector instance.

    Example:
        >>> config = {
        ...     'model_name': 'Qwen/Qwen2.5-3B-Instruct',
        ...     'temperature': 0.1,
        ...     'load_in_4bit': True
        ... }
        >>> corrector = create_corrector_from_config(config)
    """
    return LLMCorrector(**config)
