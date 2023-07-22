from typing import Optional

from transformers import AutoTokenizer

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig

DEFAULT_DIR = "OpenAssistant"
BASE_URL = f"https://api-inference.huggingface.co/models/{DEFAULT_DIR}/"
COMPLETIONS_KEY = 0
INPUT_KEY = "inputs"
TEXT_KEY = "generated_text"
AUTH_HEADER = "Authorization"
BEARER_TOKEN = "Bearer "
DEFAULT_MODEL = "oasst-sft-4-pythia-12b-epoch-3.5"
CONST_SLASH = '/'
EMPTY_STR = ''
NEWLINE = '\n'
TEMPERATURE_KEY = "temperature"
TOKENS_KEY = "max_length"


class HuggingFaceClient(BaseLLMAPIClient):
    def __init__(self, config: LLMAPIClientConfig):
        super().__init__(config)
        if self._base_url is None:
            self._base_url = BASE_URL
        if self._default_model is None:
            self._default_model = DEFAULT_MODEL
        self._headers[AUTH_HEADER] = BEARER_TOKEN + self._api_key

    async def text_completion(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 1.0,
                              model: Optional[str] = None, top_p: Optional[float] = None, **kwargs) -> list[str]:
        model = model or self._default_model
        kwargs["top_p"] = top_p
        kwargs[INPUT_KEY] = prompt
        kwargs[TEMPERATURE_KEY] = temperature
        kwargs[TOKENS_KEY] = kwargs.pop(TOKENS_KEY, max_tokens)
        response = await self._session.post(self._base_url + model + CONST_SLASH,
                                            json=kwargs,
                                            headers=self._headers,
                                            raise_for_status=True)
        response_json = await response.json()
        if isinstance(response_json, list):
            completions = response_json[COMPLETIONS_KEY][TEXT_KEY]
        else:
            completions = response_json[TEXT_KEY]
        return [completion for completion in completions.split(NEWLINE) if completion != EMPTY_STR][1:]

    def get_tokens_count(self, text: str, **kwargs) -> int:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_DIR + CONST_SLASH + self._default_model)
        return len(tokenizer.encode(text))
