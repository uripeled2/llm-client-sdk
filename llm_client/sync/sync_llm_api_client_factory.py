import asyncio

import async_to_sync
from aiohttp import ClientSession

from llm_client.llm_api_client.llm_api_client_factory import LLMAPIClientFactory, LLMAPIClientType


class SyncLLMAPIClientFactory:
    def __init__(self, ):
        self._client_session: ClientSession | None = None
        self._llm_api_client_factory: LLMAPIClientFactory = LLMAPIClientFactory()

    def __enter__(self):
        asyncio.run(self._llm_api_client_factory.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self._llm_api_client_factory.__aexit__(exc_type, exc_val, exc_tb))

    def get_llm_api_client(self, llm_api_client_type: LLMAPIClientType, **config_kwargs) -> async_to_sync.methods:
        return async_to_sync.methods(self._llm_api_client_factory.
                                     get_llm_api_client(llm_api_client_type, **config_kwargs))
