import os

from aiohttp import ClientSession

from llm_client import HuggingFaceClient, LLMAPIClientConfig

HUGGINGFACE_API_KEY = os.environ["API_KEY"]


async def main():
    async with ClientSession() as session:
        llm_client = HuggingFaceClient(LLMAPIClientConfig(HUGGINGFACE_API_KEY, session))
        text = "Who is Lebron James?"
        response = await llm_client.text_completion(text)
        token_count = llm_client.get_tokens_count(text)
        print(f"The Response Was : {response} and The Token Count Is {token_count}")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
