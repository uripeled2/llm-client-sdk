import os

from aiohttp import ClientSession

from llm_client import BaseLLMClient, OpenAIClient, AI21Client
from llm_client.llm_api_client.openai_client import ChatMessage, Role


def get_llm_api_client() -> BaseLLMClient:
    if os.getenv("API_KEY") is None:
        raise ValueError("API_KEY environment variable is not set when using api client")
    if os.getenv("LLM_CLIENT") == "openai":
        print("Using OpenAI client")
        return OpenAIClient(os.environ["API_KEY"], ClientSession(), default_model="davinci")
    elif os.getenv("LLM_CLIENT") == "ai21":
        print("Using AI21 client")
        return AI21Client(os.environ["API_KEY"], ClientSession(), default_model="j2-grande")
    else:
        raise ValueError("Unknown LLM client type")


async def main():
    llm_client = get_llm_api_client()
    text = "This is indeed a test"

    print("number of tokens:", await llm_client.get_tokens_count(text))
    if isinstance(llm_client, OpenAIClient):
        print("generated chat:", await llm_client.chat_completion(
            messages=[ChatMessage(role=Role.USER, content="Hello!")], model="gpt-3.5-turbo"))
    print("generated text:", await llm_client.text_completion(text))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
