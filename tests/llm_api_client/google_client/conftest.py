from typing import Optional

import pytest

from llm_client.llm_api_client.google_client import GoogleClient, BASE_URL, AUTH_PARAM
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig, BaseLLMAPIClient


@pytest.fixture
def model_name():
    return "text-bison-001"


@pytest.fixture
def config(client_session, model_name):
    return LLMAPIClientConfig("top-secret-api-key", client_session, default_model=model_name)


@pytest.fixture
def llm_client(config):
    return GoogleClient(config)


@pytest.fixture
def params(llm_client):
    return "?" + AUTH_PARAM + "=" + llm_client._api_key


def build_url(llm_client: BaseLLMAPIClient, path: str, model: Optional[str] = None) -> str:
    model = model or llm_client._default_model
    return BASE_URL + model + ":" + path
