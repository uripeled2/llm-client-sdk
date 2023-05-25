import pytest
import pytest_asyncio
from aiohttp import ClientSession
from aioresponses import aioresponses

from llm_client import AnthropicClient
from llm_client.llm_api_client.anthropic_client import BASE_URL, COMPLETE_PATH
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig


@pytest.fixture
def mock_aioresponse():  # TODO reuse
    with aioresponses() as m:
        yield m


@pytest_asyncio.fixture
async def client_session():
    session = ClientSession()
    yield session
    await session.close()


@pytest.fixture
def model_name():
    return "claude-v1"


@pytest.fixture
def config(client_session, model_name):
    return LLMAPIClientConfig("top-secret-api-key", client_session, default_model=model_name)


@pytest.fixture
def llm_client(config):
    return AnthropicClient(config)


@pytest.fixture
def complete_url():
    return BASE_URL + COMPLETE_PATH
