import pytest
import pytest_asyncio
from aiohttp import ClientSession
from aioresponses import aioresponses


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


@pytest_asyncio.fixture
async def client_session():
    session = ClientSession()
    yield session
    await session.close()
