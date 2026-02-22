"""OpenAI-compatible API corrector backend (future extension).

This module is a STUB. The APICorrector class defines the interface contract
and documents the intended config keys, but raises NotImplementedError on
instantiation. Implement when API-based inference is needed.

Supported APIs (once implemented):
- OpenAI (gpt-4o, gpt-3.5-turbo, etc.)
- Anthropic Claude (via openai-compatible layer)
- Local vLLM or Ollama server
- Any provider with an OpenAI-compatible /v1/chat/completions endpoint
"""

import logging
import os
from typing import Optional

from src.core.llm_corrector import BaseLLMCorrector, CorrectionResult

logger = logging.getLogger(__name__)


class APICorrector(BaseLLMCorrector):
    """LLM corrector backed by an OpenAI-compatible REST API.

    Config keys read::

        config['model']['name']              → model name for API calls (e.g. "gpt-4o")
        config['api']['base_url']            → API endpoint (e.g. "https://api.openai.com/v1")
        config['api']['api_key_env']         → env var name holding the API key
        config['api']['timeout_s']           → request timeout in seconds (default 30)
        config['api']['requests_per_minute'] → rate limit (default 60)

    Security: The API key is NEVER stored in config files. It is read from
    the environment variable named by config['api']['api_key_env'].

    Status: STUB — raises NotImplementedError on instantiation.
    Set model.backend = "transformers" and run inference on Kaggle/Colab
    until API access is available.
    """

    def __init__(self, config: dict) -> None:
        """Validate config and read API key from environment.

        Raises:
            NotImplementedError: Always — stub not yet implemented.
            ValueError: If the API key environment variable is not set
                (raised before NotImplementedError for early feedback).
        """
        api_cfg = config.get("api", {})
        key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")

        if not os.environ.get(key_env):
            raise ValueError(
                f"API key not found. Set the '{key_env}' environment variable "
                f"before using backend='api'. "
                f"Never store API keys in config files or git."
            )

        raise NotImplementedError(
            "APICorrector is not yet implemented. "
            "Use model.backend='transformers' and run inference on Kaggle/Colab. "
            "To implement: install openai>=1.0, add tenacity for retry logic, "
            "and implement correct() using client.chat.completions.create()."
        )

    def correct(
        self,
        sample_id: str,
        ocr_text: str,
        messages: list[dict],
        max_retries: int = 2,
    ) -> CorrectionResult:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError
