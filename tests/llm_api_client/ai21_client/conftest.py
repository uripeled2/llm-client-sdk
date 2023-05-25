import pytest

from llm_client.llm_api_client.ai21_client import AI21Client, COMPLETE_PATH, BASE_URL
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig


@pytest.fixture
def model_name():
    return "ada"


@pytest.fixture
def config(client_session, model_name):
    return LLMAPIClientConfig("top-secret-api-key", client_session, default_model=model_name)


@pytest.fixture
def llm_client(config):
    return AI21Client(config)


@pytest.fixture
def url(model_name):
    return build_url(model_name)


def build_url(model: str) -> str:
    return BASE_URL + model + "/" + COMPLETE_PATH
