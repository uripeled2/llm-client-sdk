import os

from aiohttp import ClientSession

from llm_client import ChatMessage, Role, OpenAIClient, LLMAPIClientConfig


OPENAI_API_KEY = os.environ["API_KEY"]
OPENAI_ORG_ID = os.getenv("ORG_ID")


async def main():
    async with ClientSession() as session:
        llm_client = OpenAIClient(LLMAPIClientConfig(OPENAI_API_KEY, session, default_model="text-davinci-003",
                                                     headers={"OpenAI-Organization": OPENAI_ORG_ID}))
        text = "This is indeed a test"

        print("number of tokens:", await llm_client.get_tokens_count(text))
        print("generated chat:", await llm_client.chat_completion(
            messages=[ChatMessage(role=Role.USER, content="Hello!")], model="gpt-3.5-turbo"))
        print("generated text:", await llm_client.text_completion(text))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
