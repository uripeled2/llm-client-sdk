from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Optional

import openai
import tiktoken
from dataclasses_json import dataclass_json, config
from tiktoken import Encoding

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig
from llm_client.consts import PROMPT_KEY

INPUT_KEY = "input"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass_json
@dataclass
class ChatMessage:
    role: Role = field(metadata=config(encoder=lambda role: role.value, decoder=Role))
    content: str
    name: Optional[str] = field(default=None, metadata=config(exclude=lambda name: name is None))


class OpenAIClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        openai.api_key = self._api_key
        openai.aiosession.set(self._session)
        self._client = openai

    async def text_completion(self, prompt: str, model: Optional[str] = None, temperature: float = 0,
                              max_tokens: int = 16, top_p: float = 1, **kwargs) -> list[str]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[PROMPT_KEY] = prompt
        kwargs["top_p"] = top_p
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
        completions = await self._client.Completion.acreate(headers=self._headers, **kwargs)
        return [choice.text for choice in completions.choices]

    async def chat_completion(self, messages: list[ChatMessage], temperature: float = 0,
                              max_tokens: int = 16, top_p: float = 1, model: Optional[str] = None, **kwargs) -> list[str]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs["messages"] = [message.to_dict() for message in messages]
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        kwargs["max_tokens"] = max_tokens
        completions = await self._client.ChatCompletion.acreate(headers=self._headers, **kwargs)
        return [choice.message.content for choice in completions.choices]

    async def embedding(self, text: str, model: Optional[str] = None, **kwargs) -> list[float]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[INPUT_KEY] = text
        embeddings = await openai.Embedding.acreate(**kwargs)
        return embeddings.data[0].embedding

    async def get_tokens_count(self, text: str, model: Optional[str] = None, **kwargs) -> int:
        if model is None:
            model = self._default_model
        return len(self._get_relevant_tokeniser(model).encode(text))

    @staticmethod
    @lru_cache(maxsize=40)
    def _get_relevant_tokeniser(model: str) -> Encoding:
        return tiktoken.encoding_for_model(model)
