from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizerBase

from llm_client import LocalClientConfig, LocalClient


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def mock_tokenizer():
    return MagicMock(PreTrainedTokenizerBase)


@pytest.fixture
def tensors_type():
    return "pt"


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def local_client(mock_model, mock_tokenizer, tensors_type, device):
    return LocalClient(LocalClientConfig(mock_model, mock_tokenizer, tensors_type, device))
