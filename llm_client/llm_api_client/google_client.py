from typing import Any, Optional

try:
    from google.ai.generativelanguage_v1beta2 import MessagePrompt
except ImportError:
    MessagePrompt = Any  # This only needed for runtime chat_completion and chat tokens count

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig
from llm_client.consts import PROMPT_KEY

COMPLETE_PATH = "generateText"
CHAT_PATH = "generateMessage"
EMBEDDING_PATH = "embedText"
TOKENIZE_PATH = "countMessageTokens"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta2/models/"
COMPLETIONS_KEY = "candidates"
TEXT_KEY = "text"
MAX_TOKENS_KEY = "maxOutputTokens"
COMPLETIONS_OUTPUT_KEY = "output"
TOKENS_KEY = "tokenCount"
MESSAGES_KEY = "messages"
MESSAGE_CONTENT_KEY = "content"
EMBEDDING_KEY = "embedding"
EMBEDDING_VALUE_KEY = "value"
AUTH_PARAM = "key"


class GoogleClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        if self._base_url is None:
            self._base_url = BASE_URL
        self._params = {AUTH_PARAM: self._api_key}

    async def text_completion(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = 64,
                              temperature: Optional[float] = None,top_p: Optional[float] = None, **kwargs) -> list[str]:
        model = model or self._default_model
        kwargs[PROMPT_KEY] = {TEXT_KEY: prompt}
        kwargs[MAX_TOKENS_KEY] = kwargs.pop(MAX_TOKENS_KEY, max_tokens)
        if top_p:
            kwargs["topP"] = top_p
        kwargs["temperature"] = kwargs.pop("temperature", temperature)
        response = await self._session.post(self._base_url + model + ":" + COMPLETE_PATH,
                                            params=self._params,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        completions = response_json[COMPLETIONS_KEY]
        return [completion[COMPLETIONS_OUTPUT_KEY] for completion in completions]

    async def embedding(self, text: str, model: Optional[str] = None, **kwargs) -> list[float]:
        model = model or self._default_model
        response = await self._session.post(self._base_url + model + ":" + EMBEDDING_PATH,
                                            params=self._params,
                                            json={TEXT_KEY: text},
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        return response_json[EMBEDDING_KEY][EMBEDDING_VALUE_KEY]

    async def get_tokens_count(self, text: str, model: Optional[str] = None,
                               messages: Optional[MessagePrompt] = None, **kwargs) -> int:
        """
        Retrieves the count of tokens in the given text using the specified model or the default_model.

        :param text: (str) The input text to tokenize and count the tokens. If the keyword argument `${MESSAGES_KEY}` \
                       is provided, this parameter will be ignored.
        :param model: (Optional[str], optional) The name of the model to use for tokenization. If not provided,
                       the default model will be used. Defaults to `None`.
        :param messages: (MessagePrompt | None, optional) The messages to tokenize and count the tokens. If provided,
                            the `text` parameter will be ignored.
        :param kwargs: Ignored.
        :return: (int) The count of tokens in the given text.
        """

        model = model or self._default_model
        if not messages:
            messages = {MESSAGES_KEY: [{MESSAGE_CONTENT_KEY: text}]}
        response = await self._session.post(self._base_url + model + ":" + TOKENIZE_PATH,
                                            params=self._params,
                                            json={PROMPT_KEY: messages},
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        return response_json[TOKENS_KEY]
