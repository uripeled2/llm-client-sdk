from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    async def text_completion(self, prompt: str, **kwargs) -> list[str]:
        raise NotImplementedError()

    async def get_tokens_count(self, text: str, **kwargs) -> int:
        raise NotImplementedError()
