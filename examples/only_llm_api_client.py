import os

from aiohttp import ClientSession

from llm_client import LLMAPIClientConfig, LLMAPIClientType, get_llm_api_client, ChatMessage, Role

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_API_CLIENT_TYPE = os.environ["LLM_API_CLIENT_TYPE"]


def get_open_ai_config(session: ClientSession) -> LLMAPIClientConfig:
    return LLMAPIClientConfig(LLM_API_KEY, session, default_model="davinci")


def get_ai21_config(session: ClientSession) -> LLMAPIClientConfig:
    return LLMAPIClientConfig(LLM_API_KEY, session, default_model="j2-grande")


async def main():
    async with ClientSession() as session:
        if LLM_API_CLIENT_TYPE == "OPEN_AI":
            llm_client = get_llm_api_client(LLMAPIClientType.OPEN_AI, get_open_ai_config(session))
        elif LLM_API_CLIENT_TYPE == "AI21":
            llm_client = get_llm_api_client(LLMAPIClientType.AI21, get_ai21_config(session))
        else:
            raise ValueError("Unknown LLM client type")

        text = "This is indeed a test"

        print("number of tokens:", await llm_client.get_tokens_count(text))
        if hasattr(llm_client, "chat_completion"):
            print("generated chat:", await llm_client.chat_completion(
                messages=[ChatMessage(role=Role.USER, content="Hello!")], model="gpt-3.5-turbo"))
        print("generated text:", await llm_client.text_completion(text))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
