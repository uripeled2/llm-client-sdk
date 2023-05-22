from enum import Enum

from aiohttp import ClientSession

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig


class LLMAPIClientType(Enum):
    OPEN_AI = "OpenAI"
    AI21 = "AI21"
    HUGGING_FACE = "HUGGING_FACE"
    ALEPH_ALPHA = "AlephAlpha"


class LLMAPIClientFactory:
    def __init__(self):
        self._client_session: ClientSession | None = None

    async def __aenter__(self):
        self._client_session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client_session.close()

    def get_llm_api_client(self, llm_api_client_type: LLMAPIClientType, **config_kwargs) -> BaseLLMAPIClient:
        if self._client_session is None:
            raise ValueError("Must be used as an context manager")
        config = LLMAPIClientConfig(session=self._client_session, **config_kwargs)
        if llm_api_client_type == LLMAPIClientType.OPEN_AI:
            from llm_client import OpenAIClient
            return OpenAIClient(config)
        elif llm_api_client_type == LLMAPIClientType.AI21:
            from llm_client import AI21Client
            return AI21Client(config)
        elif llm_api_client_type == LLMAPIClientType.HUGGING_FACE:
            from llm_client import HuggingFaceClient
            return HuggingFaceClient(config)
        elif llm_api_client_type == LLMAPIClientType.ALEPH_ALPHA:
            from llm_client import AlephAlphaClient
            return AlephAlphaClient(config)
        else:
            raise ValueError("Unknown LLM client type")
