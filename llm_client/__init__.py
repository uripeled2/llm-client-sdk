__version__ = "0.0.1"

from llm_client.llm_client_interface import LLMClientInterface

try:
    from llm_client.llm_client.openai_client import OpenAIClient
except ImportError:
    pass
try:
    from llm_client.llm_client.ai21 import AI21Client
except ImportError:
    pass
