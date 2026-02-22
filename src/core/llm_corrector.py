"""LLM corrector backends for Arabic OCR post-correction.

Defines the abstract interface (BaseLLMCorrector) and the primary
HuggingFace transformers implementation (TransformersCorrector).
A factory function (get_corrector) selects the backend from config.

All backends accept OpenAI-format message lists from PromptBuilder and
return CorrectionResult objects — the pipeline is backend-agnostic.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.data_loader import OCRSample

logger = logging.getLogger(__name__)

# Arabic Unicode range — used to validate that LLM output contains Arabic text
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CorrectionResult:
    """Result of LLM correction for a single sample.

    corrected_text is always populated:
    - On success: the LLM output
    - On failure: ocr_text (safe fallback — does not inflate error rates)
    """

    sample_id: str
    ocr_text: str
    corrected_text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    success: bool
    error: Optional[str] = None


@dataclass
class CorrectedSample:
    """OCRSample paired with its LLM correction result.

    Used throughout the Phase 2 analysis pipeline.
    ``sample.gt_text`` provides ground truth for metric calculation.
    """

    sample: "OCRSample"
    corrected_text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseLLMCorrector(ABC):
    """Abstract base class for all LLM corrector backends.

    Any backend (local transformers, OpenAI API, Anthropic, etc.) must
    implement this interface. Pipeline code depends only on this class,
    never on a concrete implementation.

    Implementations:
        TransformersCorrector — HuggingFace transformers (Kaggle/Colab)
        APICorrector          — OpenAI-compatible REST API (see api_corrector.py)

    Usage::

        corrector = get_corrector(config)
        result = corrector.correct(sample_id, ocr_text, messages)
    """

    @abstractmethod
    def correct(
        self,
        sample_id: str,
        ocr_text: str,
        messages: list[dict],
        max_retries: int = 2,
    ) -> CorrectionResult:
        """Correct a single OCR text using the provided chat messages.

        Implementations must:
        - Return a CorrectionResult in ALL cases (never raise on failure)
        - Fall back to ocr_text as corrected_text when success=False
        - Set error to a descriptive string when success=False

        Args:
            sample_id: Identifier for logging and result attribution.
            ocr_text: Original OCR text (stored in result for reference).
            messages: Chat messages list in OpenAI format (system + user).
            max_retries: Retry count on empty or failed generation.

        Returns:
            CorrectionResult with corrected_text always populated.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a string identifying the model (for metadata and logging)."""


# ---------------------------------------------------------------------------
# TransformersCorrector
# ---------------------------------------------------------------------------


class TransformersCorrector(BaseLLMCorrector):
    """LLM corrector backed by HuggingFace transformers.

    Loads model and tokenizer ONCE at initialisation. Designed to run on
    Kaggle/Colab GPU kernels. Supports optional 4-bit quantization via
    bitsandbytes for GPUs with limited VRAM.

    Config keys read::

        config['model']['name']          → HuggingFace model ID
        config['model']['temperature']   → float (default 0.1)
        config['model']['max_tokens']    → int (default 1024)
        config['model']['device']        → "auto" | "cuda" | "cpu"
        config['model']['quantize_4bit'] → bool (default False)
    """

    # Guard against OOM on unexpectedly long inputs.
    # Line-level OCR data is typically 10–80 words, well within this limit.
    MAX_INPUT_TOKENS: int = 1024

    def __init__(self, config: dict) -> None:
        """Load model and tokenizer from config.

        Args:
            config: Parsed config.yaml dict.

        Raises:
            RuntimeError: If model loading fails (e.g., model not downloaded).
            ImportError: If transformers or torch is not installed.
        """
        self._config = config
        model_cfg = config.get("model", {})

        self._model_id: str = model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507")
        self._temperature: float = float(model_cfg.get("temperature", 0.1))
        self._max_tokens: int = int(model_cfg.get("max_tokens", 1024))
        self._device: str = model_cfg.get("device", "auto")
        self._quantize_4bit: bool = bool(model_cfg.get("quantize_4bit", False))

        self._tokenizer = None
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model into memory. Called once in __init__."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for TransformersCorrector. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        logger.info("Loading tokenizer: %s", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )

        if self._quantize_4bit:
            logger.info("Loading model with 4-bit quantization (bitsandbytes)...")
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantization. "
                    "Install with: pip install bitsandbytes"
                ) from exc
        else:
            import torch as _torch
            logger.info("Loading model: %s  (device=%s)", self._model_id, self._device)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=_torch.float16,
                device_map=self._device,
                trust_remote_code=True,
            )

        self._model.eval()
        num_params = sum(p.numel() for p in self._model.parameters()) / 1e9
        logger.info("Model loaded. Parameters: %.2fB", num_params)

    @property
    def model_name(self) -> str:
        """Return the loaded model's HuggingFace ID."""
        return self._model_id

    def correct(
        self,
        sample_id: str,
        ocr_text: str,
        messages: list[dict],
        max_retries: int = 2,
    ) -> CorrectionResult:
        """Correct a single OCR text. Retries on empty output. Never raises.

        On generation failure or empty/non-Arabic output, falls back to
        returning the original ocr_text with success=False.

        Args:
            sample_id: Identifier for logging and result attribution.
            ocr_text: Original OCR text stored in result for reference.
            messages: Chat messages in OpenAI format from PromptBuilder.
            max_retries: Number of additional attempts on empty output.

        Returns:
            CorrectionResult with corrected_text always populated.
        """
        t0 = time.monotonic()
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                raw_output, prompt_tokens, output_tokens = self._generate(messages)
                corrected = self._extract_corrected_text(raw_output, ocr_text)

                if corrected == ocr_text and not raw_output.strip():
                    # Empty generation — retry
                    last_error = f"Empty generation on attempt {attempt + 1}"
                    logger.warning("[%s] %s — retrying...", sample_id, last_error)
                    continue

                return CorrectionResult(
                    sample_id=sample_id,
                    ocr_text=ocr_text,
                    corrected_text=corrected,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    latency_s=round(time.monotonic() - t0, 3),
                    success=True,
                    error=None,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning(
                    "[%s] Generation failed (attempt %d/%d): %s",
                    sample_id, attempt + 1, max_retries + 1, exc,
                )

        # All attempts exhausted — fall back to original OCR text
        logger.error(
            "[%s] All %d attempts failed. Falling back to OCR text. Last error: %s",
            sample_id, max_retries + 1, last_error,
        )
        return CorrectionResult(
            sample_id=sample_id,
            ocr_text=ocr_text,
            corrected_text=ocr_text,  # safe fallback
            prompt_tokens=0,
            output_tokens=0,
            latency_s=round(time.monotonic() - t0, 3),
            success=False,
            error=last_error,
        )

    def _generate(self, messages: list[dict]) -> tuple[str, int, int]:
        """Tokenise messages, run model.generate(), decode new tokens only.

        Args:
            messages: OpenAI-format chat messages.

        Returns:
            Tuple of (decoded_text, prompt_token_count, output_token_count).

        Raises:
            RuntimeError: If generation raises any exception.
        """
        import torch

        # Apply Qwen3 chat template — add_generation_prompt=True appends the
        # <|im_start|>assistant token so the model continues from that position.
        # enable_thinking=False disables the <think>...</think> scratchpad for
        # Qwen3 models, which keeps outputs clean and parseable.
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self._tokenizer(formatted, return_tensors="pt")

        # Move input tensors to the same device as the model
        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_len = input_ids.shape[1]

        # Guard against OOM on unexpectedly long inputs
        if prompt_len > self.MAX_INPUT_TOKENS:
            logger.warning(
                "Input %d tokens exceeds MAX_INPUT_TOKENS=%d — truncating.",
                prompt_len, self.MAX_INPUT_TOKENS,
            )
            input_ids = input_ids[:, -self.MAX_INPUT_TOKENS:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.MAX_INPUT_TOKENS:]
            prompt_len = input_ids.shape[1]

        with torch.no_grad():
            generate_kwargs: dict = dict(
                input_ids=input_ids,
                max_new_tokens=self._max_tokens,
                do_sample=True,
                temperature=self._temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            output_ids = self._model.generate(**generate_kwargs)

        # Extract only newly generated tokens (exclude the prompt)
        new_token_ids = output_ids[:, prompt_len:]
        output_tokens = new_token_ids.shape[1]

        decoded = self._tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        return decoded, prompt_len, output_tokens

    def _extract_corrected_text(self, raw_output: str, ocr_text: str) -> str:
        """Clean decoded model output into usable corrected text.

        The model is instructed to return only the corrected text. This method
        handles edge cases defensively:

        1. Strip leading/trailing whitespace and newlines
        2. If result is empty → return ocr_text (with warning)
        3. If result contains no Arabic characters → return ocr_text
           (guards against model replying in English or refusing)

        Args:
            raw_output: Decoded text from model.generate() (new tokens only).
            ocr_text: Original OCR text used as fallback.

        Returns:
            Clean corrected Arabic text string. Never empty.
        """
        cleaned = raw_output.strip()

        if not cleaned:
            logger.warning("Model returned empty output — using OCR text as fallback.")
            return ocr_text

        if not _ARABIC_RE.search(cleaned):
            logger.warning(
                "Model output contains no Arabic characters (%r…) — "
                "using OCR text as fallback.",
                cleaned[:50],
            )
            return ocr_text

        return cleaned


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_corrector(config: dict) -> BaseLLMCorrector:
    """Instantiate the corrector backend specified in config.

    Config key: ``config['model']['backend']``
        - ``"transformers"`` → TransformersCorrector (default)
        - ``"api"``          → APICorrector (from src.core.api_corrector)

    Args:
        config: Parsed config.yaml dict.

    Returns:
        Initialised BaseLLMCorrector implementation.

    Raises:
        ValueError: If the backend value is not recognised.
    """
    backend = config.get("model", {}).get("backend", "transformers")

    if backend == "transformers":
        return TransformersCorrector(config)
    elif backend == "api":
        from src.core.api_corrector import APICorrector
        return APICorrector(config)
    else:
        raise ValueError(
            f"Unknown corrector backend '{backend}'. "
            "Valid values: 'transformers', 'api'."
        )
