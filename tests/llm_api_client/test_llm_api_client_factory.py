from unittest.mock import patch

import pytest

from llm_client import get_llm_api_client, LLMAPIClientType


@pytest.mark.parametrize("client_type,client_patch",
                         [(LLMAPIClientType.OPEN_AI, "OpenAIClient"), (LLMAPIClientType.AI21, "AI21Client")])
def test_get_llm_api_client__with_client_type(client_type, client_patch):
    with patch(f"llm_client.{client_patch}") as mock_client:
        actual = get_llm_api_client(client_type, "config")

        assert actual is mock_client.return_value
        mock_client.assert_called_once_with("config")


def test_get_llm_api_client__with_unknown_client_type():
    with pytest.raises(ValueError):
        get_llm_api_client("unknown-client-type", "config")
