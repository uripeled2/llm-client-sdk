from unittest.mock import patch

import pytest

from llm_client import LLMAPIClientType, LLMAPIClientFactory, LLMAPIClientConfig
from llm_client.llm_api_client.llm_api_client_factory import get_llm_api_client_class


def test_get_llm_api_client__without_context_manager():
    llm_api_client_factory = LLMAPIClientFactory()
    with pytest.raises(ValueError):
        llm_api_client_factory.get_llm_api_client(LLMAPIClientType.OPEN_AI, api_key="super secret key")


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type,client_patch",
                         [(LLMAPIClientType.OPEN_AI, "OpenAIClient"), (LLMAPIClientType.AI21, "AI21Client"),
                          (LLMAPIClientType.HUGGING_FACE, "HuggingFaceClient"),
                          (LLMAPIClientType.ALEPH_ALPHA, "AlephAlphaClient"),
                          (LLMAPIClientType.ANTHROPIC, "AnthropicClient"),
                          (LLMAPIClientType.GOOGLE, "GoogleClient")])
async def test_get_llm_api_client__with_client_type(client_type, client_patch):
    assert len(LLMAPIClientType) == 6

    llm_api_client_factory = LLMAPIClientFactory()
    async with llm_api_client_factory:
        with patch(f"llm_client.{client_patch}") as mock_client:
            actual = llm_api_client_factory.get_llm_api_client(client_type, api_key="super secret key")

            assert actual is mock_client.return_value
            mock_client.assert_called_once_with(LLMAPIClientConfig(session=llm_api_client_factory._session,
                                                                   api_key="super secret key"))


@pytest.mark.asyncio
async def test_get_llm_api_client__with_unknown_client_type():
    llm_api_client_factory = LLMAPIClientFactory()
    async with llm_api_client_factory:
        with pytest.raises(ValueError):
            llm_api_client_factory.get_llm_api_client("unknown-client-type", api_key="super secret key")


@pytest.mark.parametrize("client_type,client_patch",
                         [(LLMAPIClientType.OPEN_AI, "OpenAIClient"), (LLMAPIClientType.AI21, "AI21Client"),
                          (LLMAPIClientType.HUGGING_FACE, "HuggingFaceClient"),
                          (LLMAPIClientType.ALEPH_ALPHA, "AlephAlphaClient"),
                          (LLMAPIClientType.ANTHROPIC, "AnthropicClient"),
                          (LLMAPIClientType.GOOGLE, "GoogleClient")])
def test_get_llm_api_client_class(client_type, client_patch):
    assert len(LLMAPIClientType) == 6

    with patch(f"llm_client.{client_patch}") as mock_client:
        actual = get_llm_api_client_class(client_type)

        assert actual is mock_client
        mock_client.assert_not_called()


def test_get_llm_api_client_class__with_unknown_client_type():
    with pytest.raises(ValueError):
        get_llm_api_client_class("unknown-client-type")
