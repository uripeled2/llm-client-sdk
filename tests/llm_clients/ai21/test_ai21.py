import pytest

from llm_client.consts import MODEL_KEY
from llm_client.llm_client.ai21 import COMPLETE_PATH, BASE_URL, DATA_KEY, TEXT_KEY, TOKENIZE_PATH, AUTH_HEADER, \
    BEARER_TOKEN
from tests.llm_clients.ai21.conftest import build_url
from tests.test_utils.load_json_resource import load_json_resource


@pytest.mark.asyncio
async def test_text_completion__sanity(mock_aioresponse, client_session, llm_client, model_name, url):
    mock_aioresponse.post(
        url,
        payload=load_json_resource("ai21/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="These are a few of my favorite")

    assert actual == [
        ' things!\n\nI love entertaining, entertaining and decorating my home, entertaining clients, entertaining '
        'friends, entertaining family...you get the point! One of my favorite things to do is plan parties']
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key},
                                             json={'prompt': 'These are a few of my favorite'},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__return_multiple_completions(mock_aioresponse, client_session, llm_client, model_name,
                                                            url):
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
                                             json={'prompt': 'These are a few of my favorite'},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__override_model(mock_aioresponse, client_session, llm_client):
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
                                             json={'prompt': 'These are a few of my favorite'},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(mock_aioresponse, client_session, llm_client, model_name, url):
    mock_aioresponse.post(
        BASE_URL + TOKENIZE_PATH,
        payload=load_json_resource("ai21/tokenize.json")
    )

    actual = await llm_client.get_tokens_count(text="These are a few of my favorite things!")

    assert actual == 3


@pytest.mark.asyncio
async def test_chat_completion__raise_not_implemented_error(llm_client):
    with pytest.raises(NotImplementedError):
        await llm_client.chat_completion(prompt="These are a few of my favorite")
