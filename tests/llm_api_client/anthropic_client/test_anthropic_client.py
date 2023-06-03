from unittest.mock import patch

import pytest

from llm_client import LLMAPIClientFactory, LLMAPIClientType, AnthropicClient
from llm_client.consts import PROMPT_KEY, MODEL_KEY
from llm_client.llm_api_client.anthropic_client import AUTH_HEADER, COMPLETIONS_KEY, MAX_TOKENS_KEY, ACCEPT_HEADER, \
    ACCEPT_VALUE


@pytest.mark.asyncio
async def test_get_llm_api_client__with_anthropic(config):
    del config.session
    async with LLMAPIClientFactory() as llm_api_client_factory:
        actual = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.ANTHROPIC, **config.__dict__)

    assert isinstance(actual, AnthropicClient)


@pytest.mark.asyncio
async def test_text_completion__sanity(mock_aioresponse, llm_client, complete_url):
    mock_aioresponse.post(
        complete_url,
        payload={COMPLETIONS_KEY: "completion text"}
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10)

    assert actual == ["completion text"]
    mock_aioresponse.assert_called_once_with(complete_url, method='POST',
                                             headers={AUTH_HEADER: llm_client._api_key,
                                                      ACCEPT_HEADER: ACCEPT_VALUE},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10, "temperature": 1,
                                                   MODEL_KEY: llm_client._default_model},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__without_max_tokens_raise_value_error(mock_aioresponse, llm_client):
    with pytest.raises(ValueError):
        await llm_client.text_completion(prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_text_completion__override_model(mock_aioresponse, llm_client, complete_url):
    new_model_name = "claude-instant"
    mock_aioresponse.post(
        complete_url,
        payload={COMPLETIONS_KEY: "completion text"}
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", model=new_model_name,
                                              max_tokens=10)

    assert actual == ["completion text"]
    mock_aioresponse.assert_called_once_with(complete_url, method='POST',
                                             headers={AUTH_HEADER: llm_client._api_key,
                                                      ACCEPT_HEADER: ACCEPT_VALUE},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10, "temperature": 1,
                                                   MODEL_KEY: new_model_name},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(mock_aioresponse, llm_client, complete_url):
    mock_aioresponse.post(
        complete_url,
        payload={COMPLETIONS_KEY: "completion text"}
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10, temperature=0.5)

    assert actual == ["completion text"]
    mock_aioresponse.assert_called_once_with(complete_url, method='POST',
                                             headers={AUTH_HEADER: llm_client._api_key,
                                                      ACCEPT_HEADER: ACCEPT_VALUE},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10,
                                                   MODEL_KEY: llm_client._default_model,
                                                   "temperature": 0.5},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(llm_client):
    with patch("llm_client.llm_api_client.anthropic_client.count_tokens") as mock_count_tokens:
        actual = await llm_client.get_tokens_count(text="These are a few of my favorite things!")

        assert actual == mock_count_tokens.return_value
        mock_count_tokens.assert_called_once_with("These are a few of my favorite things!")
