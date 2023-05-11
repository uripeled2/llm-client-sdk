from unittest.mock import patch

from llm_client import LLMAPIClientConfig, get_llm_client, LocalClientConfig, LLMAPIClientConfigWithClientType, \
    LLMAPIClientType


def test_get_llm_client__with_llm_api_client_config():
    with patch("llm_client.llm_client_factory.get_llm_api_client") as mock_get_llm_api_client:
        config = LLMAPIClientConfigWithClientType(LLMAPIClientType.OPEN_AI,
                                                  LLMAPIClientConfig("api-key", "session", "base-url", "default-model",
                                                                     {"header": "value"}))

        actual = get_llm_client(config)

        assert actual is mock_get_llm_api_client.return_value
        mock_get_llm_api_client.assert_called_once_with(config.llm_api_client_type, config.llm_api_client_config)


def test_get_llm_client__with_local_client_config():
    with patch("llm_client.LocalClient") as mock_local_client:
        config = LocalClientConfig("model", "tokenizer", "tensors-type", "device", {"header": "value"})

        actual = get_llm_client(config)

        assert actual is mock_local_client.return_value
        mock_local_client.assert_called_once_with(config)
