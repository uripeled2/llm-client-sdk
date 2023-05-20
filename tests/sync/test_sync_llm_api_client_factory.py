from unittest.mock import patch

from llm_client import SyncLLMAPIClientFactory, LLMAPIClientFactory


def test_contex_manger():
    with patch.object(LLMAPIClientFactory, "__aenter__") as mock_aenter:
        with patch.object(LLMAPIClientFactory, "__aexit__") as mock_aexit:
            with SyncLLMAPIClientFactory():
                pass

            mock_aenter.assert_called_once_with()
            mock_aexit.assert_called_once_with(None, None, None)


def test_get_llm_api_client():
    with patch.object(LLMAPIClientFactory,  "get_llm_api_client") as mock_get_llm_api_client:
        with patch("llm_client.sync.sync_llm_api_client_factory.get_sync_llm_client") as mock_get_sync_llm_client:
            with SyncLLMAPIClientFactory() as sync_llm_api_client_factory:
                actual = sync_llm_api_client_factory.get_llm_api_client("llm_api_client_type", api_key="api-key")

            assert actual == mock_get_sync_llm_client.return_value
            mock_get_llm_api_client.assert_called_once_with("llm_api_client_type", api_key="api-key")
            mock_get_sync_llm_client.assert_called_once_with(mock_get_llm_api_client.return_value)