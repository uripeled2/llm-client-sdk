from typing import Optional

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig
from llm_client.consts import PROMPT_KEY

COMPLETE_PATH = "complete"
TOKENIZE_PATH = "tokenize"
EMBEDDING_PATH = "semantic_embed"
BASE_URL = "https://api.aleph-alpha.com/"
COMPLETIONS_KEY = "completions"
TEXT_KEY = "completion"
TOKENS_IDS_KEY = "token_ids"
TOKENS_KEY = "tokens"
REPRESENTATION_KEY = "representation"
REPRESENTATION_DEFAULT_VALUE = "symmetric"
EMBEDDING_KEY = "embedding"
AUTH_HEADER = "Authorization"
BEARER_TOKEN = "Bearer "
MAX_TOKENS_KEY = "maximum_tokens"


class AlephAlphaClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        if self._base_url is None:
            self._base_url = BASE_URL
        self._headers[AUTH_HEADER] = BEARER_TOKEN + self._api_key

    async def text_completion(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                              temperature: float = 0,top_p: float = 0, **kwargs) -> \
            list[str]:
        self._set_model_in_kwargs(kwargs, model)
        if max_tokens is None:
            raise ValueError("max_tokens must be specified")
        kwargs[PROMPT_KEY] = prompt
        kwargs["top_p"] = top_p
        kwargs["maximum_tokens"] = kwargs.pop("maximum_tokens", max_tokens)
        kwargs["temperature"] = temperature
        response = await self._session.post(self._base_url + COMPLETE_PATH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        completions = response_json[COMPLETIONS_KEY]
        return [completion[TEXT_KEY] for completion in completions]

    async def embedding(self, text: str, model: Optional[str] = None,
                        representation: str = REPRESENTATION_DEFAULT_VALUE,
                        **kwargs) -> list[float]:
        self._set_model_in_kwargs(kwargs, model)
        kwargs[REPRESENTATION_KEY] = representation
        kwargs[PROMPT_KEY] = text
        response = await self._session.post(self._base_url + EMBEDDING_PATH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        return response_json[EMBEDDING_KEY]

    async def get_tokens_count(self, text: str, model: Optional[str] = None, **kwargs) -> int:
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
