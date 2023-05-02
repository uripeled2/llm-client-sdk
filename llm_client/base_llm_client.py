from abc import ABC, abstractmethod

from llm_client.consts import MODEL_KEY


class BaseLLMClient(ABC):
    def __init__(self, default_model: str | None = None):
        self._default_model = default_model

    @abstractmethod
    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        raise NotImplementedError()

    async def chat_completion(self, model: str | None = None, **kwargs) -> list[str]:
        raise NotImplementedError()

    async def get_tokens_count(self, text: str, *args) -> int:
        raise NotImplementedError()

    def _set_model_in_kwargs(self, kwargs, model: str | None) -> None:
        if model is not None:
            kwargs[MODEL_KEY] = model
        kwargs.setdefault(MODEL_KEY, self._default_model)
