from aiohttp import ClientSession

from llm_client.consts import PROMPT_KEY
from llm_client.llm_client_interface import LLMClientInterface


COMPLETE_PATH = "complete"
TOKENIZE_PATH = "tokenize"
BASE_URL = "https://api.ai21.com/studio/v1/"
COMPLETIONS_KEY = "completions"
DATA_KEY = "data"
TEXT_KEY = "text"
TOKENS_KEY = "tokens"
AUTH_HEADER = "Authorization"
BEARER_TOKEN = "Bearer "


class AI21Client(LLMClientInterface):

    def __init__(self, api_key: str, session: ClientSession, default_model: str | None = None,
                 base_url: str = BASE_URL):
        super().__init__(default_model)
        self._api_key = api_key
        self._session = session
        self._base_url = base_url

    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        model = model or self._default_model
        kwargs[PROMPT_KEY] = prompt
        response = await self._session.post(self._base_url + model + "/" + COMPLETE_PATH,
                                            json=kwargs,
                                            headers={AUTH_HEADER: BEARER_TOKEN + self._api_key},
                                            raise_for_status=True)
        response_json = await response.json()
        completions = response_json[COMPLETIONS_KEY]
        return [completion[DATA_KEY][TEXT_KEY] for completion in completions]

    async def get_tokens_count(self, text: str, *args) -> int:
        response = await self._session.post(self._base_url + TOKENIZE_PATH,
                                            json={TEXT_KEY: text},
                                            headers={AUTH_HEADER: BEARER_TOKEN + self._api_key},
                                            raise_for_status=True)
        response_json = await response.json()
        return len(response_json[TOKENS_KEY])
