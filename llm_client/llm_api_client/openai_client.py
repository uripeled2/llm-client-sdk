from functools import lru_cache
from typing import Optional

import openai
import tiktoken
from tiktoken import Encoding

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage
from llm_client.consts import PROMPT_KEY

INPUT_KEY = "input"
MODEL_NAME_TO_TOKENS_PER_MESSAGE_AND_TOKENS_PER_NAME = {
    "gpt-3.5-turbo-0613": (3, 1),
    "gpt-3.5-turbo-16k-0613": (3, 1),
    "gpt-4-0314": (3, 1),
    "gpt-4-32k-0314": (3, 1),
    "gpt-4-0613": (3, 1),
    "gpt-4-32k-0613": (3, 1),
    # every message follows <|start|>{role/name}\n{content}<|end|>\n, if there's a name, the role is omitted
    "gpt-3.5-turbo-0301": (4, -1),
}


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
                              max_tokens: int = 16, top_p: float = 1, model: Optional[str] = None, **kwargs) \
            -> list[str]:
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

    async def get_chat_tokens_count(self, messages: list[ChatMessage], model: Optional[str] = None) -> int:
        """
        This is based on:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        model = self._get_model_name_for_tokeniser(model)
        encoding = self._get_relevant_tokeniser(model)
        tokens_per_message, tokens_per_name = MODEL_NAME_TO_TOKENS_PER_MESSAGE_AND_TOKENS_PER_NAME[model]
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message.content))
            num_tokens += len(encoding.encode(message.role.value))
            if message.name:
                num_tokens += len(encoding.encode(message.name))
                num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _get_model_name_for_tokeniser(self, model: Optional[str] = None) -> str:
        if model is None:
            model = self._default_model
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            return model
        elif model == "gpt-3.5-turbo-0301":
            return model
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning tokeniser assuming gpt-3.5-turbo-0613.")
            return "gpt-3.5-turbo-0613"
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning tokeniser assuming gpt-4-0613.")
            return "gpt-4-0613"
        else:
            raise NotImplementedError(
                f"""not implemented for model {model}. 
                See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    @staticmethod
    @lru_cache(maxsize=40)
    def _get_relevant_tokeniser(model: str) -> Encoding:
        return tiktoken.encoding_for_model(model)
