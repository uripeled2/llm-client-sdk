from enum import Enum

from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig


class LLMAPIClientType(Enum):
    OPEN_AI = "OpenAI"
    AI21 = "AI21"
    HUGGING_FACE = "HUGGING_FACE"


def get_llm_api_client(llm_api_client_type: LLMAPIClientType, config: LLMAPIClientConfig) -> BaseLLMAPIClient:
    if llm_api_client_type == LLMAPIClientType.OPEN_AI:
        from llm_client import OpenAIClient
        return OpenAIClient(config)
    elif llm_api_client_type == LLMAPIClientType.AI21:
        from llm_client import AI21Client
        return AI21Client(config)
    elif llm_api_client_type == LLMAPIClientType.HUGGING_FACE:
        from llm_client import HuggingFaceClient
        return HuggingFaceClient(config)
    else:
        raise ValueError("Unknown LLM client type")
