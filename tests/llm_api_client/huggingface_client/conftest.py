import pytest

from llm_client.llm_api_client.huggingface_client import HuggingFaceClient, BASE_URL
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig


@pytest.fixture
def model_name():
    return "oasst-sft-4-pythia-12b-epoch-3.5"


@pytest.fixture
def config(client_session, model_name):
    return LLMAPIClientConfig("top-secret-api-key", client_session, default_model=model_name)


@pytest.fixture
def llm_client(config):
    return HuggingFaceClient(config)


@pytest.fixture
def url(model_name):
    return build_url(model_name)


def build_url(model: str) -> str:
    return BASE_URL + model + "/"
