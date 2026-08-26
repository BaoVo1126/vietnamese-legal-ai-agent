from __future__ import annotations
from ..config import Settings, get_settings
from ..logging_config import get_logger
from .base import BaseLLMClient

logger = get_logger(__name__)


class VLLMClient(BaseLLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, timeout: float = 120.0) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, *, task: str = "generic",
                 temperature: float | None = None, max_tokens: int | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM task=%s -> %d ký tự", task, len(content))
        return content


def build_llm_client(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.llm_backend == "openai_compatible":
        try:
            return VLLMClient(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                              model=settings.llm_model, temperature=settings.llm_temperature,
                              max_tokens=settings.llm_max_tokens,
                              timeout=settings.llm_timeout_seconds)
        except Exception as error:
            logger.warning("Không khởi tạo được LLM client (%s) - dùng RuleBasedStubLLM.",
                           error)
    from .stub_client import RuleBasedStubLLM

    logger.info("Dùng RuleBasedStubLLM (profile MVP, chạy offline).")
    return RuleBasedStubLLM()
