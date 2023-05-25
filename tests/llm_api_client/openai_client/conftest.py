from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientSession

from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig
from llm_client.llm_api_client.openai_client import OpenAIClient


@pytest.fixture
def model_name():
    return "ada"


@pytest.fixture
def openai_mock():
    with patch("llm_client.llm_api_client.openai_client.openai") as openai_mock:
        yield openai_mock


@pytest.fixture
def config(model_name):
    return LLMAPIClientConfig("fake-api-key", MagicMock(ClientSession), default_model=model_name)


@pytest.fixture
def open_ai_client(config):
    return OpenAIClient(config)


@pytest.fixture
def tiktoken_mock():
    with patch("llm_client.llm_api_client.openai_client.tiktoken") as tiktoken_mock:
        yield tiktoken_mock
