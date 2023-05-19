import os

from aiohttp import ClientSession
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from llm_client import LLMAPIClientConfig, LLMAPIClientType, get_llm_api_client, ChatMessage, Role, LocalClient, \
    LocalClientConfig

LLM_CLIENT_TYPE = os.environ["LLM_CLIENT_TYPE"]
# api client env var
LLM_API_KEY = os.getenv("LLM_API_KEY")
# local client env var
MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH")
TENSORS_TYPE = os.getenv("TENSORS_TYPE")
DEVICE = os.getenv("DEVICE")


def get_local_client_config() -> LocalClientConfig:
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
    except ValueError:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    return LocalClientConfig(model, tokenizer, TENSORS_TYPE, DEVICE)


def get_open_ai_config(session: ClientSession) -> LLMAPIClientConfig:
    return LLMAPIClientConfig(LLM_API_KEY, session, default_model="davinci")


def get_ai21_config(session: ClientSession) -> LLMAPIClientConfig:
    return LLMAPIClientConfig(LLM_API_KEY, session, default_model="j2-grande")


def get_huggingface_config(session: ClientSession) -> LLMAPIClientConfig:
    return LLMAPIClientConfig(LLM_API_KEY, session)


async def main():
    if LLM_CLIENT_TYPE == "LOCAL":
        llm_client = LocalClient(get_local_client_config())
    elif LLM_CLIENT_TYPE == "OPEN_AI":
        llm_client = get_llm_api_client(LLMAPIClientType.OPEN_AI, get_open_ai_config(ClientSession()))
    elif LLM_CLIENT_TYPE == "AI21":
        llm_client = get_llm_api_client(LLMAPIClientType.AI21, get_ai21_config(ClientSession()))
    elif LLM_CLIENT_TYPE == "HUGGING_FACE":
        llm_client = get_llm_api_client(LLMAPIClientType.HUGGING_FACE, get_huggingface_config(ClientSession()))
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
