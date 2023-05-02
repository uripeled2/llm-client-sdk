from functools import lru_cache

import openai
import tiktoken
from tiktoken import Encoding

from llm_client.consts import PROMPT_KEY
from llm_client.base_llm_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    def __init__(self, client: openai, default_model: str | None = None):
        super().__init__(default_model)
        self._client = client

    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[PROMPT_KEY] = prompt
        completions = await self._client.Completion.acreate(**kwargs)
        return [choice.text for choice in completions.choices]

    async def chat_completion(self, model: str | None = None, **kwargs) -> list[str]:
        self._set_model_in_kwargs(kwargs, model)
        completions = await self._client.ChatCompletion.acreate(**kwargs)
        return [choice.message.content for choice in completions.choices]

    async def get_tokens_count(self, text: str, model: str | None = None) -> int:
        if model is None:
            model = self._default_model
        return len(self._get_relevant_tokeniser(model).encode(text))

    @staticmethod
    @lru_cache(maxsize=40)
    def _get_relevant_tokeniser(model: str) -> Encoding:
        return tiktoken.encoding_for_model(model)
