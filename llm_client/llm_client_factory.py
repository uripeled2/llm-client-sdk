from dataclasses import dataclass

from llm_client import BaseLLMClient
from llm_client.llm_api_client.base_llm_api_client import LLMAPIClientConfig
from llm_client.llm_api_client.llm_api_client_factory import get_llm_api_client, LLMAPIClientType
from llm_client.llm_client.local_client import LocalClientConfig


@dataclass
class LLMAPIClientConfigWithClientType:
    llm_api_client_type: LLMAPIClientType
    llm_api_client_config: LLMAPIClientConfig


def get_llm_client(llm_api_client_config: LLMAPIClientConfigWithClientType | LocalClientConfig) -> BaseLLMClient:
    if isinstance(llm_api_client_config, LLMAPIClientConfigWithClientType):
        return get_llm_api_client(llm_api_client_config.llm_api_client_type,
                                  llm_api_client_config.llm_api_client_config)
    elif isinstance(llm_api_client_config, LocalClientConfig):
        from llm_client import LocalClient
        return LocalClient(llm_api_client_config)
    else:
        raise TypeError("bad type for llm_api_client_config, must be LLMAPIClientConfig or LocalClientConfig")
