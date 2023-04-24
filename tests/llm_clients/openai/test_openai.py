from unittest.mock import AsyncMock, patch

import pytest
from openai.openai_object import OpenAIObject

from tests.test_utils.load_json_resource import load_json_resource


@pytest.mark.asyncio
async def test_text_completion__sanity(openai_mock, open_ai_client, model_name):
    openai_mock.Completion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/text_completion.json")))
    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nThis is indeed a test"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_text_completion__return_multiple_completions(openai_mock, open_ai_client, model_name):
    open_ai_object = OpenAIObject.construct_from(load_json_resource("openai/text_completion.json"))
    open_ai_object.choices.append(OpenAIObject.construct_from({"text": "second completion"}))
    openai_mock.Completion.acreate = AsyncMock(return_value=open_ai_object)

    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nThis is indeed a test", "second completion"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_text_completion__override_model(openai_mock, open_ai_client, model_name):
    new_model_name = "gpt3"
    openai_mock.Completion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/text_completion.json")))

    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite", model=new_model_name)

    assert actual == ["\n\nThis is indeed a test"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=new_model_name,
        prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_chat_completion__sanity(openai_mock, open_ai_client, model_name):
    openai_mock.ChatCompletion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json")))

    actual = await open_ai_client.chat_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nHello there, how may I assist you today?"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_chat_completion__return_multiple_completions(openai_mock, open_ai_client, model_name):
    open_ai_object = OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json"))
    open_ai_object.choices.append(OpenAIObject.construct_from({"message": {"content": "second completion"}}))
    openai_mock.ChatCompletion.acreate = AsyncMock(return_value=open_ai_object)

    actual = await open_ai_client.chat_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nHello there, how may I assist you today?", "second completion"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_chat_completion__override_model(openai_mock, open_ai_client, model_name):
    new_model_name = "gpt3"
    openai_mock.ChatCompletion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json")))

    actual = await open_ai_client.chat_completion(prompt="These are a few of my favorite", model=new_model_name)

    assert actual == ["\n\nHello there, how may I assist you today?"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=new_model_name,
        prompt="These are a few of my favorite")


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(model_name, open_ai_client):
    with patch("LLMClient.llm_clients.openai_client.tiktoken") as tiktoken_mock:
        tokeniser_mock = tiktoken_mock.encoding_for_model.return_value
        tokeniser_mock.encode.return_value = [123, 456]
        text = "This is a test"

        actual = await open_ai_client.get_tokens_count(text=text)

        assert actual == len(tokeniser_mock.encode.return_value)
        tiktoken_mock.encoding_for_model.assert_called_once_with(model_name)
        tokeniser_mock.encode.assert_called_once_with(text)


@pytest.mark.asyncio
async def test_get_tokens_count__override_model(open_ai_client):
    with patch("LLMClient.llm_clients.openai_client.tiktoken") as tiktoken_mock:
        tokeniser_mock = tiktoken_mock.encoding_for_model.return_value
        tokeniser_mock.encode.return_value = [123, 456]
        text = "This is a test"
        model_name = "gpt3"

        actual = await open_ai_client.get_tokens_count(text=text, model=model_name)

        assert actual == len(tokeniser_mock.encode.return_value)
        tiktoken_mock.encoding_for_model.assert_called_once_with(model_name)
        tokeniser_mock.encode.assert_called_once_with(text)
