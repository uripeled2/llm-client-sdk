from typing import Any, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm_client import BaseLLMClient


class LocalClient(BaseLLMClient):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 tensors_type: str, device: str, encode_kwargs: Optional[dict[str, Any]] = None):
        if not model.can_generate():
            raise TypeError(f"{model} is not a text generation model")

        self._model: PreTrainedModel = model
        self._tokenizer: PreTrainedTokenizerBase = tokenizer
        self._tensors_type: str = tensors_type
        self._device: str = device
        self._encode_kwargs: dict[str, Any] = encode_kwargs or {}
        self._encode_kwargs["return_tensors"] = tensors_type

    async def text_completion(self, prompt: str, **kwargs) -> list[str]:
        input_ids = self._encode(prompt)
        outputs = self._model.generate(input_ids, **kwargs)
        return [self._tokenizer.decode(output) for output in outputs]

    async def get_tokens_count(self, text: str, **kwargs) -> int:
        return len(self._encode(text))

    def _encode(self, prompt: str) -> list[int]:
        return self._tokenizer.encode(prompt, **self._encode_kwargs).to(self._device)
