from aiohttp import ClientSession

from llm_client.base_llm_api_client import BaseLLMAPIClient
from llm_client.consts import PROMPT_KEY


COMPLETE_PATH = "complete"
TOKENIZE_PATH = "tokenize"
BASE_URL = "https://api.ai21.com/studio/v1/"
COMPLETIONS_KEY = "completions"
DATA_KEY = "data"
TEXT_KEY = "text"
TOKENS_KEY = "tokens"
AUTH_HEADER = "Authorization"
BEARER_TOKEN = "Bearer "


class AI21Client(BaseLLMAPIClient):
    def __init__(self, api_key: str, session: ClientSession, base_url: str | None = None,
                 default_model: str | None = None, **headers):
        super().__init__(api_key, session, base_url or BASE_URL, default_model, **headers)
        self._headers[AUTH_HEADER] = BEARER_TOKEN + self._api_key

    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        model = model or self._default_model
        kwargs[PROMPT_KEY] = prompt
        response = await self._session.post(self._base_url + model + "/" + COMPLETE_PATH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        completions = response_json[COMPLETIONS_KEY]
        return [completion[DATA_KEY][TEXT_KEY] for completion in completions]

    async def get_tokens_count(self, text: str, **kwargs) -> int:
        response = await self._session.post(self._base_url + TOKENIZE_PATH,
                                            json={TEXT_KEY: text},
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        return len(response_json[TOKENS_KEY])
