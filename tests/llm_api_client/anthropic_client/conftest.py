from unittest.mock import patch, AsyncMock

import pytest

from llm_client.llm_api_client.anthropic_client import BASE_URL, COMPLETE_PATH, VERSION_HEADER, AnthropicClient
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig


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


@pytest.fixture
def number_of_tokens():
    return 10


@pytest.fixture
def anthropic_version():
    return "2023-06-01"


@pytest.fixture(autouse=True)
def mock_anthropic(number_of_tokens, anthropic_version):
    with patch("llm_client.llm_api_client.anthropic_client.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.count_tokens = AsyncMock(return_value=number_of_tokens)
        mock_anthropic.return_value.default_headers = {VERSION_HEADER: anthropic_version}
        yield mock_anthropic
