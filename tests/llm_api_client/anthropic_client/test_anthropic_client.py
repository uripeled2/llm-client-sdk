from unittest.mock import AsyncMock

import pytest

from llm_client import LLMAPIClientFactory, LLMAPIClientType, ChatMessage
from llm_client.consts import PROMPT_KEY, MODEL_KEY
from llm_client.llm_api_client.anthropic_client import AUTH_HEADER, COMPLETIONS_KEY, MAX_TOKENS_KEY, ACCEPT_HEADER, \
    ACCEPT_VALUE, VERSION_HEADER, AnthropicClient, USER_PREFIX, ASSISTANT_PREFIX, START_PREFIX, SYSTEM_START_PREFIX, \
    SYSTEM_END_PREFIX
from llm_client.llm_api_client.base_llm_api_client import Role


@pytest.mark.asyncio
async def test_get_llm_api_client__with_anthropic(config):
    del config.session
    async with LLMAPIClientFactory() as llm_api_client_factory:
        actual = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.ANTHROPIC, **config.__dict__)

    assert isinstance(actual, AnthropicClient)

@pytest.mark.asyncio
async def test_chat_completion_sanity(llm_client):
    text_completion_mock = AsyncMock(return_value=["completion text"])
    llm_client.text_completion = text_completion_mock

    actual = await llm_client.chat_completion(messages=[ChatMessage(Role.USER, "Why is the sky blue?")], max_tokens=10)

    assert actual == ["completion text"]
    text_completion_mock.assert_awaited_once_with(f"{START_PREFIX}{USER_PREFIX} Why is the sky blue?"
                                                  f"{START_PREFIX}{ASSISTANT_PREFIX}", None, 10, 1)


@pytest.mark.asyncio
async def test_chat_completion_with_assistant_in_the_end(llm_client):
    text_completion_mock = AsyncMock(return_value=["completion text"])
    llm_client.text_completion = text_completion_mock

    actual = await llm_client.chat_completion(messages=[ChatMessage(Role.USER, "Why is the sky blue?"),
                                                        ChatMessage(Role.ASSISTANT, "Answer - ")], temperature=10)

    assert actual == ["completion text"]
    text_completion_mock.assert_awaited_once_with(f"{START_PREFIX}{USER_PREFIX} Why is the sky blue?"
                                                  f"{START_PREFIX}{ASSISTANT_PREFIX} Answer -", None, None,
                                                  10)


@pytest.mark.asyncio
async def test_chat_completion_with_system(llm_client):
    text_completion_mock = AsyncMock(return_value=["completion text"])
    llm_client.text_completion = text_completion_mock

    actual = await llm_client.chat_completion(messages=[ChatMessage(Role.SYSTEM, "Be nice!"),
                                                        ChatMessage(Role.USER, "Why is the sky blue?")], max_tokens=10,
                                              temperature=2)

    assert actual == ["completion text"]
    text_completion_mock.assert_awaited_once_with(f"{START_PREFIX}{USER_PREFIX} "
                                                  f"{SYSTEM_START_PREFIX}Be nice!{SYSTEM_END_PREFIX}{START_PREFIX}"
                                                  f"{USER_PREFIX} Why is the sky blue?"
                                                  f"{START_PREFIX}{ASSISTANT_PREFIX}", None, 10, 2)


@pytest.mark.asyncio
async def test_text_completion__sanity(mock_aioresponse, llm_client, complete_url, anthropic_version):
    mock_aioresponse.post(
        complete_url,
        payload={COMPLETIONS_KEY: "completion text"}
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10,)

    assert actual == ["completion text"]
    mock_aioresponse.assert_called_once_with(complete_url, method='POST',
                                             headers={AUTH_HEADER: llm_client._api_key,
                                                      ACCEPT_HEADER: ACCEPT_VALUE,
                                                      VERSION_HEADER: anthropic_version},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10, "temperature": 1,
                                                   MODEL_KEY: llm_client._default_model},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__with_version_header(mock_aioresponse, config, complete_url):
    mock_aioresponse.post(
        complete_url,
        payload={COMPLETIONS_KEY: "completion text"}
    )
    config.headers[VERSION_HEADER] = "1.0.0"
    llm_client = AnthropicClient(config)

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10)

    assert actual == ["completion text"]
    mock_aioresponse.assert_called_once_with(complete_url, method='POST',
                                             headers={AUTH_HEADER: llm_client._api_key,
                                                      ACCEPT_HEADER: ACCEPT_VALUE,
                                                      VERSION_HEADER: "1.0.0"},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10, "temperature": 1,
                                                   MODEL_KEY: llm_client._default_model},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__without_max_tokens_raise_value_error(mock_aioresponse, llm_client):
    with pytest.raises(ValueError):
        await llm_client.text_completion(prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_text_completion__override_model(mock_aioresponse, llm_client, complete_url, anthropic_version):
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
                                                      ACCEPT_HEADER: ACCEPT_VALUE,
                                                      VERSION_HEADER: anthropic_version},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10, "temperature": 1,
                                                   MODEL_KEY: new_model_name},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(mock_aioresponse, llm_client, complete_url, anthropic_version):
    mock_aioresponse.post(
        complete_url,
        payload={COMPLETIONS_KEY: "completion text"}
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10, temperature=0.5,top_p=0.5)

    assert actual == ["completion text"]
    mock_aioresponse.assert_called_once_with(complete_url, method='POST',
                                             headers={AUTH_HEADER: llm_client._api_key,
                                                      ACCEPT_HEADER: ACCEPT_VALUE,
                                                      VERSION_HEADER: anthropic_version},
                                             json={PROMPT_KEY: 'These are a few of my favorite',
                                                   MAX_TOKENS_KEY: 10,
                                                   MODEL_KEY: llm_client._default_model,
                                                   "temperature": 0.5, "top_p" : 0.5},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(llm_client, number_of_tokens, mock_anthropic):
    actual = await llm_client.get_tokens_count(text="These are a few of my favorite things!")

    assert actual == 10
    mock_anthropic.return_value.count_tokens.assert_awaited_once_with("These are a few of my favorite things!")
