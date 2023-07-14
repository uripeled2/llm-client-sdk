import pytest

from llm_client import LLMAPIClientType, LLMAPIClientFactory
from llm_client.llm_api_client.ai21_client import BASE_URL, DATA_KEY, TEXT_KEY, TOKENIZE_PATH, AUTH_HEADER, \
    BEARER_TOKEN, AI21Client
from tests.llm_api_client.ai21_client.conftest import build_url
from tests.test_utils.load_json_resource import load_json_resource


@pytest.mark.asyncio
async def test_get_llm_api_client__with_ai21(config):
    del config.session
    async with LLMAPIClientFactory() as llm_api_client_factory:
        actual = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.AI21, **config.__dict__)

    assert isinstance(actual, AI21Client)


@pytest.mark.asyncio
async def test_text_completion__sanity(mock_aioresponse, llm_client, url):
    mock_aioresponse.post(
        url,
        payload=load_json_resource("ai21/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite")

    assert actual == [
        ' things!\n\nI love entertaining, entertaining and decorating my home, entertaining clients, entertaining '
        'friends, entertaining family...you get the point! One of my favorite things to do is plan parties']
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key },
                                             json={'prompt': 'These are a few of my favorite', "maxTokens" : 16, "temperature" : 0.7, "topP" : 1 },
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__return_multiple_completions(mock_aioresponse, llm_client, url):
    payload = load_json_resource("ai21/text_completion.json")
    payload["completions"].append({DATA_KEY: {TEXT_KEY: "second completion"}})
    mock_aioresponse.post(url, payload=payload)

    actual = await llm_client.text_completion(prompt="These are a few of my favorite")

    assert actual == [
        ' things!\n\nI love entertaining, entertaining and decorating my home, entertaining clients, entertaining '
        'friends, entertaining family...you get the point! One of my favorite things to do is plan parties',
        "second completion"
    ]
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key},
                                             json={'prompt': 'These are a few of my favorite', "maxTokens" : 16, "temperature" : 0.7, "topP" : 1  },
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__override_model(mock_aioresponse, llm_client):
    new_model_name = "gpt3"
    url = build_url(new_model_name)
    mock_aioresponse.post(
        url,
        payload=load_json_resource("ai21/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", model=new_model_name)

    assert actual == [
        ' things!\n\nI love entertaining, entertaining and decorating my home, entertaining clients, entertaining '
        'friends, entertaining family...you get the point! One of my favorite things to do is plan parties']
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key},
                                             json={'prompt': 'These are a few of my favorite', "maxTokens" : 16, "temperature" : 0.7, "topP" : 1 },
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(mock_aioresponse, llm_client, url):
    mock_aioresponse.post(
        url,
        payload=load_json_resource("ai21/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10)

    assert actual == [
        ' things!\n\nI love entertaining, entertaining and decorating my home, entertaining clients, entertaining '
        'friends, entertaining family...you get the point! One of my favorite things to do is plan parties']
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key},
                                             json={'prompt': 'These are a few of my favorite', "maxTokens" : 10, "temperature" : 0.7 ,"topP" : 1},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(mock_aioresponse, llm_client, url):
    mock_aioresponse.post(
        BASE_URL + TOKENIZE_PATH,
        payload=load_json_resource("ai21/tokenize.json")
    )

    actual = await llm_client.get_tokens_count(text="These are a few of my favorite things!")

    assert actual == 3
