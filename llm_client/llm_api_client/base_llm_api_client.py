from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from aiohttp import ClientSession
except ImportError:
    ClientSession = Any

from llm_client import BaseLLMClient
from llm_client.consts import MODEL_KEY


@dataclass
class LLMAPIClientConfig:
    api_key: str
    session: ClientSession
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    headers: dict[str, Any] = field(default_factory=dict)


class BaseLLMAPIClient(BaseLLMClient, ABC):
    def __init__(self, config: LLMAPIClientConfig):
        self._api_key: str = config.api_key
        self._session: ClientSession = config.session
        self._base_url: str = config.base_url
        self._default_model: str = config.default_model
        self._headers: dict[str, str] = config.headers

    @abstractmethod
    async def text_completion(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None, **kwargs) -> list[str]:
        raise NotImplementedError()

    async def embedding(self, text: str, model: Optional[str] = None, **kwargs) -> list[float]:
        raise NotImplementedError()

    def _set_model_in_kwargs(self, kwargs, model: Optional[str]) -> None:
        if model is not None:
            kwargs[MODEL_KEY] = model
        kwargs.setdefault(MODEL_KEY, self._default_model)
