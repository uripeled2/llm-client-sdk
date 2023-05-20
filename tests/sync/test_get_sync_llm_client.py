from unittest.mock import patch

from llm_client.sync.get_sync_llm_client import get_sync_llm_client


def test_sanity():
    with patch("llm_client.sync.get_sync_llm_client.async_to_sync") as mock_async_to_sync:
        actual = get_sync_llm_client("llm_client")

        assert actual == mock_async_to_sync.methods.return_value
        mock_async_to_sync.methods.assert_called_once_with("llm_client")
