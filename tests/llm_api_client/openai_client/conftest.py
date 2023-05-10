from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import ClientSession

from llm_client.llm_api_client.openai_client import OpenAIClient


@pytest.fixture
def model_name():
    return "ada"


@pytest_asyncio.fixture
async def client_session():
    session = ClientSession()
    yield session
    await session.close()


@pytest.fixture
def openai_mock():
    with patch("llm_client.llm_api_client.openai_client.openai") as openai_mock:
        yield openai_mock


@pytest.fixture
def open_ai_client(model_name):
    return OpenAIClient("fake-api-key", MagicMock(ClientSession), default_model=model_name)


@pytest.fixture
def tiktoken_mock():
    with patch("llm_client.llm_api_client.openai_client.tiktoken") as tiktoken_mock:
        yield tiktoken_mock