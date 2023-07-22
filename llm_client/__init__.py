__version__ = "0.8.0"

from llm_client.base_llm_client import BaseLLMClient

# load api clients
try:
    from llm_client.llm_api_client.base_llm_api_client import BaseLLMAPIClient, LLMAPIClientConfig, ChatMessage, Role
    from llm_client.llm_api_client.llm_api_client_factory import LLMAPIClientFactory, LLMAPIClientType
    # load base-api clients
    try:
        from llm_client.llm_api_client.ai21_client import AI21Client
        from llm_client.llm_api_client.aleph_alpha_client import AlephAlphaClient
        from llm_client.llm_api_client.google_client import GoogleClient, MessagePrompt
    except ImportError:
        pass
    # load apis with different dependencies
    try:
        from llm_client.llm_api_client.openai_client import OpenAIClient
    except ImportError:
        pass
    try:
        from llm_client.llm_api_client.huggingface_client import HuggingFaceClient
    except ImportError:
        pass
    try:
        from llm_client.llm_api_client.anthropic_client import AnthropicClient
    except ImportError:
        pass
except ImportError:
    pass
# load local clients
try:
    from llm_client.llm_client.local_client import LocalClient, LocalClientConfig
except ImportError:
    pass
# load sync support
try:
    from llm_client.sync.sync_llm_api_client_factory import init_sync_llm_api_client
except ImportError:
    pass
