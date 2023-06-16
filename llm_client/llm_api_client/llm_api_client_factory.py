from enum import Enum
from typing import Optional

from aiohttp import ClientSession

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig


class LLMAPIClientType(Enum):
    OPEN_AI = "OpenAI"
    AI21 = "AI21"
    HUGGING_FACE = "HUGGING_FACE"
    ALEPH_ALPHA = "AlephAlpha"
    ANTHROPIC = "ANTHROPIC"
    GOOGLE = "GOOGLE"


class LLMAPIClientFactory:
    def __init__(self):
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        self._session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()

    def get_llm_api_client(self, llm_api_client_type: LLMAPIClientType, **config_kwargs) -> BaseLLMAPIClient:
        if self._session is None:
            raise ValueError("Must be used as an context manager")
        config = LLMAPIClientConfig(session=self._session, **config_kwargs)
        llm_api_client_class = get_llm_api_client_class(llm_api_client_type)
        return llm_api_client_class(config)


def get_llm_api_client_class(llm_api_client_type: LLMAPIClientType):
    if llm_api_client_type == LLMAPIClientType.OPEN_AI:
        from llm_client import OpenAIClient
        return OpenAIClient
    elif llm_api_client_type == LLMAPIClientType.AI21:
        from llm_client import AI21Client
        return AI21Client
    elif llm_api_client_type == LLMAPIClientType.HUGGING_FACE:
        from llm_client import HuggingFaceClient
        return HuggingFaceClient
    elif llm_api_client_type == LLMAPIClientType.ALEPH_ALPHA:
        from llm_client import AlephAlphaClient
        return AlephAlphaClient
    elif llm_api_client_type == LLMAPIClientType.ANTHROPIC:
        from llm_client import AnthropicClient
        return AnthropicClient
    elif llm_api_client_type == LLMAPIClientType.GOOGLE:
        from llm_client import GoogleClient
        return GoogleClient
    else:
        raise ValueError("Unknown LLM client type")
