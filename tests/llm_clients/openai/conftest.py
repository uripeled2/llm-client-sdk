from unittest.mock import MagicMock

import openai
import pytest

from LLMClient.llm_clients.openai_client import OpenAIClient


@pytest.fixture
def model_name():
    return "ada"


@pytest.fixture
def openai_mock():
    return MagicMock(openai)


@pytest.fixture
def open_ai_client(openai_mock, model_name):
    return OpenAIClient(openai_mock, model_name)
