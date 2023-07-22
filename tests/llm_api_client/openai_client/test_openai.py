from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientSession
from openai.openai_object import OpenAIObject

from llm_client import OpenAIClient, LLMAPIClientType, LLMAPIClientFactory
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig, Role
from llm_client.llm_api_client.openai_client import ChatMessage
from tests.test_utils.load_json_resource import load_json_resource


@pytest.mark.asyncio
async def test_get_llm_api_client__with_open_ai(config):
    del config.session
    async with LLMAPIClientFactory() as llm_api_client_factory:
        actual = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.OPEN_AI, **config.__dict__)

    assert isinstance(actual, OpenAIClient)


def test_init__sanity(openai_mock, client_session):
    OpenAIClient(LLMAPIClientConfig("fake_api_key", client_session))

    assert openai_mock.api_key == "fake_api_key"
    openai_mock.aiosession.set.assert_called_once()


@pytest.mark.asyncio
async def test_text_completion__sanity(openai_mock, open_ai_client, model_name):
    openai_mock.Completion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/text_completion.json")))
    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nThis is indeed a test"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite",
        headers={}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_text_completion__return_multiple_completions(openai_mock, open_ai_client, model_name):
    open_ai_object = OpenAIObject.construct_from(load_json_resource("openai/text_completion.json"))
    open_ai_object.choices.append(OpenAIObject.construct_from({"text": "second completion"}))
    openai_mock.Completion.acreate = AsyncMock(return_value=open_ai_object)

    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nThis is indeed a test", "second completion"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite",
        headers={}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_text_completion__override_model(openai_mock, open_ai_client, model_name):
    new_model_name = "gpt3"
    openai_mock.Completion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/text_completion.json")))

    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite", model=new_model_name)

    assert actual == ["\n\nThis is indeed a test"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=new_model_name,
        prompt="These are a few of my favorite",
        headers={}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(openai_mock, open_ai_client, model_name):
    openai_mock.Completion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/text_completion.json")))

    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite", max_tokens=10)

    assert actual == ["\n\nThis is indeed a test"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite",
        temperature=0, max_tokens=10, top_p=1,
        headers={})


@pytest.mark.asyncio
async def test_text_completion__with_headers(openai_mock, model_name):
    openai_mock.Completion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/text_completion.json")))
    open_ai_client = OpenAIClient(LLMAPIClientConfig("fake_api_key", MagicMock(ClientSession), default_model=model_name,
                                                     headers={"header_name": "header_value"}))

    actual = await open_ai_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["\n\nThis is indeed a test"]
    openai_mock.Completion.acreate.assert_awaited_once_with(
        model=model_name,
        prompt="These are a few of my favorite",
        headers={"header_name": "header_value"}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_chat_completion__sanity(openai_mock, open_ai_client, model_name):
    openai_mock.ChatCompletion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json")))

    actual = await open_ai_client.chat_completion([ChatMessage(Role.USER, "Hello!")])

    assert actual == ["\n\nHello there, how may I assist you today?"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=model_name,
        messages=[{'content': 'Hello!', 'role': 'user'}],
        headers={}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_chat_completion__return_multiple_completions(openai_mock, open_ai_client, model_name):
    open_ai_object = OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json"))
    open_ai_object.choices.append(OpenAIObject.construct_from({"message": {"content": "second completion"}}))
    openai_mock.ChatCompletion.acreate = AsyncMock(return_value=open_ai_object)

    actual = await open_ai_client.chat_completion([ChatMessage(Role.USER, "Hello!")])

    assert actual == ["\n\nHello there, how may I assist you today?", "second completion"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=model_name,
        messages=[{'content': 'Hello!', 'role': 'user'}],
        headers={}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_chat_completion__override_model(openai_mock, open_ai_client, model_name):
    new_model_name = "gpt3"
    openai_mock.ChatCompletion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json")))

    actual = await open_ai_client.chat_completion([ChatMessage(Role.USER, "Hello!")], model=new_model_name)

    assert actual == ["\n\nHello there, how may I assist you today?"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=new_model_name,
        messages=[{'content': 'Hello!', 'role': 'user'}],
        headers={}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
async def test_chat_completion__with_kwargs(openai_mock, open_ai_client, model_name):
    openai_mock.ChatCompletion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json")))

    actual = await open_ai_client.chat_completion([ChatMessage(Role.USER, "Hello!")], max_tokens=10, top_p=1)

    assert actual == ["\n\nHello there, how may I assist you today?"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=model_name,
        messages=[{'content': 'Hello!', 'role': 'user'}],
        max_tokens=10,
        headers={}, temperature=0, top_p=1)


@pytest.mark.asyncio
async def test_chat_completion__with_headers(openai_mock, model_name):
    openai_mock.ChatCompletion.acreate = AsyncMock(
        return_value=OpenAIObject.construct_from(load_json_resource("openai/chat_completion.json")))
    open_ai_client = OpenAIClient(LLMAPIClientConfig("fake_api_key", MagicMock(ClientSession), default_model=model_name,
                                                     headers={"header_name": "header_value"}))

    actual = await open_ai_client.chat_completion([ChatMessage(Role.USER, "Hello!")])

    assert actual == ["\n\nHello there, how may I assist you today?"]
    openai_mock.ChatCompletion.acreate.assert_awaited_once_with(
        model=model_name,
        messages=[{'content': 'Hello!', 'role': 'user'}],
        headers={"header_name": "header_value"}, temperature=0, max_tokens=16, top_p=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name,expected", [("gpt-3.5-turbo-0301", 127), ("gpt-3.5-turbo-0613", 129),
                                                 ("gpt-3.5-turbo", 129), ("gpt-4-0314", 129), ("gpt-4-0613", 129),
                                                 ("gpt-4", 129)])
async def test_get_chat_tokens_count__with_examples_from_openai_cookbook(model_name, expected, open_ai_client):
    example_messages = [
        ChatMessage(Role.SYSTEM,
                    "You are a helpful, pattern-following assistant that translates corporate jargon "
                    "into plain English."),
        ChatMessage(Role.SYSTEM, "New synergies will help drive top-line growth.", name="example_user"),
        ChatMessage(Role.SYSTEM, "Things working well together will increase revenue.", name="example_assistant"),
        ChatMessage(Role.SYSTEM,
                    "Let's circle back when we have more bandwidth to touch base on opportunities "
                    "for increased leverage.", name="example_user"),
        ChatMessage(Role.SYSTEM, "Let's talk later when we're less busy about how to do better.",
                    name="example_assistant"),
        ChatMessage(Role.USER,
                    "This late pivot means we don't have time to boil the ocean for the client deliverable."),
    ]

    actual = await open_ai_client.get_chat_tokens_count(example_messages, model=model_name)

    assert actual == expected


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(model_name, open_ai_client, tiktoken_mock):
    tokeniser_mock = tiktoken_mock.encoding_for_model.return_value
    tokeniser_mock.encode.return_value = [123, 456]
    text = "This is a test"

    actual = await open_ai_client.get_tokens_count(text=text)

    assert actual == len(tokeniser_mock.encode.return_value)
    tiktoken_mock.encoding_for_model.assert_called_once_with(model_name)
    tokeniser_mock.encode.assert_called_once_with(text)


@pytest.mark.asyncio
async def test_get_tokens_count__override_model(open_ai_client, tiktoken_mock):
    tokeniser_mock = tiktoken_mock.encoding_for_model.return_value
    tokeniser_mock.encode.return_value = [123, 456]
    text = "This is a test"
    model_name = "gpt3"

    actual = await open_ai_client.get_tokens_count(text=text, model=model_name)

    assert actual == len(tokeniser_mock.encode.return_value)
    tiktoken_mock.encoding_for_model.assert_called_once_with(model_name)
    tokeniser_mock.encode.assert_called_once_with(text)
