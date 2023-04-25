from unittest.mock import MagicMock, AsyncMock

import pytest
import pytest_asyncio
from aiohttp import BasicAuth, ClientSession
from aioresponses import aioresponses

from llm_client.llm_client.ai21 import AI21Client, COMPLETE_PATH, BASE_URL


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


@pytest_asyncio.fixture
async def client_session():
    session = ClientSession()
    yield session
    await session.close()


@pytest.fixture
def model_name():
    return "ada"


@pytest.fixture
def llm_client(client_session, model_name):
    return AI21Client(BasicAuth("username", "password"), client_session, model_name)


@pytest.fixture
def url(llm_client, model_name):
    return BASE_URL + model_name + COMPLETE_PATH
