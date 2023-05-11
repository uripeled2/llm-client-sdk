from dataclasses import dataclass, field
from typing import Any

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except ImportError:
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any

from llm_client import BaseLLMClient


@dataclass
class LocalClientConfig:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    tensors_type: str
    device: str
    encode_kwargs: dict[str, Any] = field(default_factory=dict)


class LocalClient(BaseLLMClient):
    def __init__(self, llm_client_config: LocalClientConfig):
        if not llm_client_config.model.can_generate():
            raise TypeError(f"{llm_client_config.model} is not a text generation model")

        self._model: PreTrainedModel = llm_client_config.model
        self._tokenizer: PreTrainedTokenizerBase = llm_client_config.tokenizer
        self._tensors_type: str = llm_client_config.tensors_type
        self._device: str = llm_client_config.device
        self._encode_kwargs: dict[str, Any] = llm_client_config.encode_kwargs
        self._encode_kwargs["return_tensors"] = llm_client_config.tensors_type

    async def text_completion(self, prompt: str, **kwargs) -> list[str]:
        input_ids = self._encode(prompt)
        outputs = self._model.generate(input_ids, **kwargs)
        return [self._tokenizer.decode(output) for output in outputs]

    async def get_tokens_count(self, text: str, **kwargs) -> int:
        return len(self._encode(text))

    def _encode(self, prompt: str) -> list[int]:
        return self._tokenizer.encode(prompt, **self._encode_kwargs).to(self._device)
