import pytest

from llm_client import LLMAPIClientType, LLMAPIClientFactory
from llm_client.llm_api_client.huggingface_client import AUTH_HEADER, \
    BEARER_TOKEN, HuggingFaceClient
from tests.test_utils.load_json_resource import load_json_resource


@pytest.mark.asyncio
async def test_get_llm_api_client__with_hugging_face(config):
    del config.session
    async with LLMAPIClientFactory() as llm_api_client_factory:

        actual = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.HUGGING_FACE, **config.__dict__)

    assert isinstance(actual, HuggingFaceClient)


@pytest.mark.asyncio
async def test_text_completion__sanity(mock_aioresponse, llm_client, url):
    mock_aioresponse.post(
        url,
        payload=load_json_resource("huggingface/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="who is kobe bryant")

    assert actual == ['Kobe Bryant is a retired professional basketball player who played for the Los Angeles Lakers of']
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key},
                                             json={'inputs': 'who is kobe bryant',"max_length": None, "temperature": 1.0, "top_p" : None},
                                             raise_for_status=True)


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(mock_aioresponse, llm_client, url):
    mock_aioresponse.post(
        url,
        payload=load_json_resource("huggingface/text_completion.json")
    )

    actual = await llm_client.text_completion(prompt="who is kobe bryant",max_tokens = 10)

    assert actual == ['Kobe Bryant is a retired professional basketball player who played for the Los Angeles Lakers of']
    mock_aioresponse.assert_called_once_with(url, method='POST',
                                             headers={AUTH_HEADER: BEARER_TOKEN + llm_client._api_key},
                                             json={'inputs': 'who is kobe bryant',"max_length": 10, "temperature": 1.0, "top_p" : None},
                                             raise_for_status=True)


@pytest.mark.asyncio
def test_get_tokens_count__sanity(mock_aioresponse, llm_client, url):
    actual = llm_client.get_tokens_count(text="is queen elisabeth alive?")
    assert actual == 7
