from typing import Optional

from anthropic import AsyncAnthropic

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage, Role
from llm_client.consts import PROMPT_KEY

COMPLETE_PATH = "complete"
BASE_URL = "https://api.anthropic.com/v1/"
COMPLETIONS_KEY = "completion"
AUTH_HEADER = "x-api-key"
ACCEPT_HEADER = "Accept"
VERSION_HEADER = "anthropic-version"
ACCEPT_VALUE = "application/json"
MAX_TOKENS_KEY = "max_tokens_to_sample"
USER_PREFIX = "Human:"
ASSISTANT_PREFIX = "Assistant:"
START_PREFIX = "\n\n"
SYSTEM_START_PREFIX = "<admin>"
SYSTEM_END_PREFIX = "</admin>"


class AnthropicClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        if self._base_url is None:
            self._base_url = BASE_URL
        self._anthropic = AsyncAnthropic()
        if self._headers.get(VERSION_HEADER) is None:
            self._headers[VERSION_HEADER] = self._anthropic.default_headers[VERSION_HEADER]
        self._headers[ACCEPT_HEADER] = ACCEPT_VALUE
        self._headers[AUTH_HEADER] = self._api_key

    async def chat_completion(self, messages: list[ChatMessage], model: Optional[str] = None,
                              max_tokens: Optional[int] = None, temperature: float = 1, **kwargs) -> list[str]:
        return await self.text_completion(self.messages_to_text(messages), model, max_tokens, temperature, **kwargs)

    async def text_completion(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None,
                              temperature: float = 1, top_p: Optional[float] = None,
                              **kwargs) -> \
            list[str]:
        if max_tokens is None and kwargs.get(MAX_TOKENS_KEY) is None:
            raise ValueError(f"max_tokens or {MAX_TOKENS_KEY} must be specified")
        if top_p:
            kwargs["top_p"] = top_p
        self._set_model_in_kwargs(kwargs, model)
        kwargs[PROMPT_KEY] = prompt
        kwargs[MAX_TOKENS_KEY] = kwargs.pop(MAX_TOKENS_KEY, max_tokens)
        kwargs["temperature"] = temperature
        response = await self._session.post(self._base_url + COMPLETE_PATH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        return [response_json[COMPLETIONS_KEY]]

    async def get_chat_tokens_count(self, messages: list[ChatMessage], **kwargs) -> int:
        return await self.get_tokens_count(self.messages_to_text(messages), **kwargs)

    async def get_tokens_count(self, text: str, **kwargs) -> int:
        return await self._anthropic.count_tokens(text)

    def messages_to_text(self, messages: list[ChatMessage]) -> str:
        prompt = START_PREFIX
        prompt += START_PREFIX.join(map(self._message_to_prompt, messages))
        if messages[-1].role != Role.ASSISTANT:
            prompt += START_PREFIX
            prompt += self._message_to_prompt(ChatMessage(role=Role.ASSISTANT, content=""))
        return prompt.rstrip()

    @staticmethod
    def _message_to_prompt(message: ChatMessage) -> str:
        if message.role == Role.USER:
            return f"{USER_PREFIX} {message.content}"
        if message.role == Role.ASSISTANT:
            return f"{ASSISTANT_PREFIX} {message.content}"
        if message.role == Role.SYSTEM:
            return f"{USER_PREFIX} {SYSTEM_START_PREFIX}{message.content}{SYSTEM_END_PREFIX}"
        raise ValueError(f"Unknown role: {message.role}")
