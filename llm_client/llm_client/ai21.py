from aiohttp import ClientSession, BasicAuth

from LLMClient.consts import MODEL_KEY, PROMPT_KEY
from LLMClient.llm_client_interface import LLMClientInterface


COMPLETE_PATH = "/complete"
TOKENIZE_PATH = "/tokenize"
BASE_URL = "https://api.ai21.com/studio/v1/"
COMPLETIONS_KEY = "completions"
DATA_KEY = "data"
TEXT_KEY = "text"
TOKENS_KEY = "tokens"


class AI21Client(LLMClientInterface):

    def __init__(self, auth: BasicAuth, session: ClientSession, default_model: str | None = None,
                 base_url: str = BASE_URL):
        super().__init__(default_model)
        self._auth = auth
        self._session = session
        self._base_url = base_url

    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[PROMPT_KEY] = prompt
        response = await self._session.post(self._base_url + kwargs[MODEL_KEY] + COMPLETE_PATH, auth=self._auth,
                                            data=kwargs,
                                            raise_for_status=True)
        response_json = await response.json()
        completions = response_json[COMPLETIONS_KEY]
        return [completion[DATA_KEY][TEXT_KEY] for completion in completions]

    async def get_tokens_count(self, text: str, *args) -> int:
        response = await self._session.post(self._base_url + TOKENIZE_PATH, auth=self._auth,
                                            data={TEXT_KEY: text},
                                            raise_for_status=True)
        response_json = await response.json()
        return len(response_json[TOKENS_KEY])
