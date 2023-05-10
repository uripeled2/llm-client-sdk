from abc import ABC, abstractmethod

from aiohttp import ClientSession

from llm_client import BaseLLMClient
from llm_client.consts import MODEL_KEY


class BaseLLMAPIClient(BaseLLMClient, ABC):
    def __init__(self, api_key: str, session: ClientSession, base_url: str | None = None,
                 default_model: str | None = None, **headers):
        self._api_key: str = api_key
        self._session: ClientSession = session
        self._base_url: str = base_url
        self._default_model: str = default_model
        self._headers: dict[str, str] = headers

    @abstractmethod
    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        raise NotImplementedError()

    def _set_model_in_kwargs(self, kwargs, model: str | None) -> None:
        if model is not None:
            kwargs[MODEL_KEY] = model
        kwargs.setdefault(MODEL_KEY, self._default_model)
