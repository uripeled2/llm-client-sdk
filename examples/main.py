import os
from dataclasses import dataclass
from typing import Type

import openai
from aiohttp import ClientSession

from llm_client import BaseLLMClient, OpenAIClient, AI21Client


@dataclass
class LLMClientConfig:
    client: Type[BaseLLMClient]
    kwargs: dict[str, str] = None


def _get_openai_config() -> LLMClientConfig:
    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set when using OpenAI client")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return LLMClientConfig(
        client=OpenAIClient,
        kwargs={"client": openai, "default_model": "davinci"},
    )


def _get_ai21_config() -> LLMClientConfig:
    if os.getenv("AI21_API_KEY") is None:
        raise ValueError("AI21_API_KEY environment variable is not set when using AI21 client")
    bearer_token = os.getenv("AI21_API_KEY")
    return LLMClientConfig(
        client=AI21Client,
        kwargs={"bearer_token": bearer_token, "session": ClientSession(), "default_model": "j2-grande"},
    )


def _get_llm_client_from_config(config: LLMClientConfig) -> BaseLLMClient:
    return config.client(**config.kwargs)


def get_llm_client() -> BaseLLMClient:
    if os.getenv("LLM_CLIENT") == "openai":
        print("Using OpenAI client")
        return _get_llm_client_from_config(_get_openai_config())
    elif os.getenv("LLM_CLIENT") == "ai21":
        print("Using AI21 client")
        return _get_llm_client_from_config(_get_ai21_config())
    else:
        raise ValueError("Unknown LLM client type")


async def main():
    llm_client = get_llm_client()
    text = "This is indeed a test"
    print("number of tokens:", await llm_client.get_tokens_count(text))
    print("generated text:", await llm_client.text_completion(text))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
