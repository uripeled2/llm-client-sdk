import pytest

from llm_client import LLMAPIClientType, LLMAPIClientFactory
from llm_client.consts import PROMPT_KEY
from llm_client.llm_api_client.google_client import TEXT_KEY, GoogleClient, COMPLETE_PATH, AUTH_PARAM, CHAT_PATH, \
    EMBEDDING_PATH, TOKENIZE_PATH, MESSAGES_KEY, MESSAGE_CONTENT_KEY, MAX_TOKENS_KEY
from tests.llm_api_client.google_client.conftest import build_url
from tests.test_utils.load_json_resource import load_json_resource


@pytest.mark.asyncio
async def test_get_llm_api_client__with_google_client(config):
    del config.session
    async with LLMAPIClientFactory() as llm_api_client_factory:
        actual = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.GOOGLE, **config.__dict__)

    assert isinstance(actual, GoogleClient)


@pytest.mark.asyncio
async def test_text_completion__sanity(mock_aioresponse, llm_client, params):
    url = build_url(llm_client, COMPLETE_PATH)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ['Once upon a time, there was a young girl named Lily...',
                      'Once upon a time, there was a young boy named Billy...']
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={PROMPT_KEY: {TEXT_KEY: 'These are a few of my favorite'},
                                                   MAX_TOKENS_KEY: 64,
                                                   'temperature': None},
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_text_completion__override_model(mock_aioresponse, llm_client, params):
    new_model_name = "text-bison-002"
    url = build_url(llm_client, COMPLETE_PATH, new_model_name)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", model=new_model_name)

    assert actual == ['Once upon a time, there was a young girl named Lily...',
                      'Once upon a time, there was a young boy named Billy...']
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={PROMPT_KEY: {TEXT_KEY: 'These are a few of my favorite'},
                                                   MAX_TOKENS_KEY: 64,
                                                   'temperature': None},
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(mock_aioresponse, llm_client, params):
    url = build_url(llm_client, COMPLETE_PATH)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite", max_tokens=10, blabla="aaa", top_p= 0.95)

    assert actual == ['Once upon a time, there was a young girl named Lily...',
                      'Once upon a time, there was a young boy named Billy...']
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={PROMPT_KEY: {TEXT_KEY: 'These are a few of my favorite'},
                                                   MAX_TOKENS_KEY: 10,
                                                   'temperature': None,
                                                   'blabla': 'aaa',"topP" : 0.95},
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_embedding__sanity(mock_aioresponse, llm_client, params):
    url = build_url(llm_client, EMBEDDING_PATH)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/embedding.json")
    )

    actual = await llm_client.embedding(text="These are a few of my favorite")

    assert actual == [0.0011238843, -0.040586308, -0.013174802, 0.015497498]
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={TEXT_KEY: 'These are a few of my favorite'},
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_embedding__override_model(mock_aioresponse, llm_client, params):
    new_model_name = "text-bison-002"
    url = build_url(llm_client, EMBEDDING_PATH, new_model_name)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/embedding.json")
    )

    actual = await llm_client.embedding(text="These are a few of my favorite", model=new_model_name)

    assert actual == [0.0011238843, -0.040586308, -0.013174802, 0.015497498]
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={TEXT_KEY: 'These are a few of my favorite'},
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_embedding__with_kwargs_not_pass_through(mock_aioresponse, llm_client, params):
    url = build_url(llm_client, EMBEDDING_PATH)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/embedding.json")
    )

    actual = await llm_client.embedding(text="These are a few of my favorite", max_tokens=10)

    assert actual == [0.0011238843, -0.040586308, -0.013174802, 0.015497498]
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={TEXT_KEY: 'These are a few of my favorite'},
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(mock_aioresponse, llm_client, params):
    url = build_url(llm_client, TOKENIZE_PATH)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/tokens_count.json")
    )

    actual = await llm_client.get_tokens_count(text="These are a few of my favorite")

    assert actual == 23
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={
                                                 PROMPT_KEY: {
                                                     MESSAGES_KEY: [
                                                         {MESSAGE_CONTENT_KEY: "These are a few of my favorite"},
                                                     ]
                                                 }
                                             },
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_get_tokens_count__override_model(mock_aioresponse, llm_client, params):
    new_model_name = "text-bison-002"
    url = build_url(llm_client, TOKENIZE_PATH, new_model_name)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/tokens_count.json")
    )

    actual = await llm_client.get_tokens_count(text="These are a few of my favorite", model=new_model_name)

    assert actual == 23
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={
                                                 PROMPT_KEY: {
                                                     MESSAGES_KEY: [
                                                         {MESSAGE_CONTENT_KEY: "These are a few of my favorite"},
                                                     ]
                                                 }
                                             },
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )


@pytest.mark.asyncio
async def test_get_tokens_count__kwargs_not_pass_through(mock_aioresponse, llm_client, params):
    url = build_url(llm_client, TOKENIZE_PATH)
    mock_aioresponse.post(
        url + params,
        payload=load_json_resource("google/tokens_count.json")
    )

    actual = await llm_client.get_tokens_count(text="These are a few of my favorite", max_tokens=10)

    assert actual == 23
    mock_aioresponse.assert_called_once_with(url, method='POST', params={AUTH_PARAM: llm_client._api_key},
                                             json={
                                                 PROMPT_KEY: {
                                                     MESSAGES_KEY: [
                                                         {MESSAGE_CONTENT_KEY: "These are a few of my favorite"},
                                                     ]
                                                 }
                                             },
                                             headers=llm_client._headers,
                                             raise_for_status=True,
                                             )
