from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig
from llm_client.consts import PROMPT_KEY

COMPLETE_PATH = "complete"
TOKENIZE_PATH = "tokenize"
BASE_URL = "https://api.aleph-alpha.com/"
COMPLETIONS_KEY = "completions"
TEXT_KEY = "completion"
TOKENS_IDS_KEY = "token_ids"
TOKENS_KEY = "tokens"
AUTH_HEADER = "Authorization"
BEARER_TOKEN = "Bearer "
MAX_TOKENS_KEY = "maximum_tokens"


class AlephAlphaClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        if self._base_url is None:
            self._base_url = BASE_URL
        self._headers[AUTH_HEADER] = BEARER_TOKEN + self._api_key

    async def text_completion(self, prompt: str, model: str | None = None, max_tokens: int | None = None, **kwargs) ->\
            list[str]:
        if max_tokens is None:
            raise ValueError("max_tokens must be specified")
        self._set_model_in_kwargs(kwargs, model)
        kwargs[PROMPT_KEY] = prompt
        kwargs[MAX_TOKENS_KEY] = max_tokens
        response = await self._session.post(self._base_url + COMPLETE_PATH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        completions = response_json[COMPLETIONS_KEY]
        return [completion[TEXT_KEY] for completion in completions]

    async def get_tokens_count(self, text: str, model: str | None = None, **kwargs) -> int:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[TOKENS_KEY] = False
        kwargs[TOKENS_IDS_KEY] = True
        kwargs[PROMPT_KEY] = text
        response = await self._session.post(self._base_url + TOKENIZE_PATH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        return len(response_json[TOKENS_IDS_KEY])
