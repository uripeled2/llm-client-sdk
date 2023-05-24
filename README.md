# LLM-Client-SDK
[![Test](https://github.com/uripeled2/llm-client-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/uripeled2/llm-client-sdk/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/github/license/uripeled2/llm-client-sdk.svg)](https://opensource.org/licenses/MIT)

LLM-Client-SDK is an SDK for communicating with generative AI large language models
(We currently support - OpenAI, AI21, HuggingfaceHub, Aleph Alpha,
Local models with transformers - and many more soon).

Our vision is to provide async native and production ready SDK while creating 
a powerful and fast integration with different LLM without letting the user lose 
any flexibility (API params, endpoints etc.). *We also provide sync version, see
more details below in Usage section.

## Base Interface
The package expose two simple interface for communicating with LLMs (In the future we 
will expend the interface to support more tasks like embeddings, list models, edits, etc.
and we will add a standardized for LLMs param like max_tokens, temperature, etc.):
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from aiohttp import ClientSession


class BaseLLMClient(ABC):
    @abstractmethod
    async def text_completion(self, prompt: str, **kwargs) -> list[str]:
        raise NotImplementedError()

    async def get_tokens_count(self, text: str, **kwargs) -> int:
        raise NotImplementedError()



@dataclass
class LLMAPIClientConfig:
    api_key: str
    session: ClientSession
    base_url: str | None = None
    default_model: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)


class BaseLLMAPIClient(BaseLLMClient, ABC):
    def __init__(self, config: LLMAPIClientConfig):
        ...

    @abstractmethod
    async def text_completion(self, prompt: str, model: str | None = None, **kwargs) -> list[str]:
        raise NotImplementedError()

    async def embedding(self, text: str, model: str | None = None, **kwargs) -> list[float]:
        raise NotImplementedError()
```

## Requirements

Python 3.10+

## Installation
If you are worried about the size of the package you can install only the clients you need,
by default we install none of the clients.

For all current clients support
```console
$ pip install llm-client[all]
```
For only the base interface and some light LLMs clients (AI21 and Aleph Alpha)
```console
$ pip install llm-client
```
### Optional Dependencies
For all current api clients support
```console
$ pip install llm-client[api]
```
For only local client support
```console
$ pip install llm-client[local]
```
For sync support
```console
$ pip install llm-client[sync]
```
For only OpenAI support
```console
$ pip install llm-client[openai]
```
For only HuggingFace support
```console
$ pip install llm-client[huggingface]
```


## Usage

Using OpenAI directly through OpenAIClient - Maximum control and best practice in production
```python
import os
from aiohttp import ClientSession
from llm_client import ChatMessage, Role, OpenAIClient, LLMAPIClientConfig

OPENAI_API_KEY = os.environ["API_KEY"]
OPENAI_ORG_ID = os.getenv("ORG_ID")


async def main():
    async with ClientSession() as session:
        llm_client = OpenAIClient(LLMAPIClientConfig(OPENAI_API_KEY, session, default_model="text-davinci-003",
                                                     headers={"OpenAI-Organization": OPENAI_ORG_ID}))  # The headers are optional
        text = "This is indeed a test"

        print("number of tokens:", await llm_client.get_tokens_count(text))  # 5
        print("generated chat:", await llm_client.chat_completion(  
            messages=[ChatMessage(role=Role.USER, content="Hello!")], model="gpt-3.5-turbo"))  # ['Hi there! How can I assist you today?']
        print("generated text:", await llm_client.text_completion(text))  # [' string\n\nYes, this is a test string. Test strings are used to']
        print("generated embedding:", await llm_client.embedding(text))  # [0.0023064255, -0.009327292, ...]
```
Using LLMAPIClientFactory - Perfect if you want to move fast and to not handle the client session yourself
```python
import os
from llm_client import LLMAPIClientFactory, LLMAPIClientType

OPENAI_API_KEY = os.environ["API_KEY"]


async def main():
    async with LLMAPIClientFactory() as llm_api_client_factory:
        llm_client = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.OPEN_AI,
                                                               api_key=OPENAI_API_KEY)

        await llm_client.text_completion(prompt="This is indeed a test")
        await llm_client.text_completion(prompt="This is indeed a test", max_length=50)

        
# Or if you don't want to use async
from llm_client import SyncLLMAPIClientFactory

with SyncLLMAPIClientFactory() as llm_api_client_factory:
    llm_client = llm_api_client_factory.get_llm_api_client(LLMAPIClientType.OPEN_AI,
                                                           api_key=OPENAI_API_KEY)

    llm_client.text_completion(prompt="This is indeed a test")
    llm_client.text_completion(prompt="This is indeed a test", max_length=50)
```
Local model
```python
import os
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from llm_client import LocalClientConfig, LocalClient

async def main():
    try:
        model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
    except ValueError:
        model = AutoModelForSeq2SeqLM.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
    llm_client = LocalClient(LocalClientConfig(model, tokenizer, os.environ["TENSORS_TYPE"], os.environ["DEVICE"]))

    await llm_client.text_completion(prompt="This is indeed a test")
    await llm_client.text_completion(prompt="This is indeed a test", max_length=50)


# Or if you don't want to use async
import async_to_sync

try:
    model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
except ValueError:
    model = AutoModelForSeq2SeqLM.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
llm_client = LocalClient(LocalClientConfig(model, tokenizer, os.environ["TENSORS_TYPE"], os.environ["DEVICE"]))

llm_client = async_to_sync.methods(llm_client)

llm_client.text_completion(prompt="This is indeed a test")
llm_client.text_completion(prompt="This is indeed a test", max_length=50)
```

## Contributing

Contributions are welcome! Please check out the todos below, and feel free to open issue or a pull request.

### Todo
*The list is unordered*

- [ ] Add support for more LLMs
- [x] Add support for more functions via LLMs 
  - [x] embeddings
  - [ ] list models
  - [ ] edits
  - [ ] more
- [ ] Add contributing guidelines
- [ ] Create an easy way to run multiple LLMs in parallel with the same prompts
- [ ] Convert common models parameter (e.g. temperature, max_tokens, etc.)

### Development
To install the package in development mode, run the following command:
```console
$ pip install -e ".[all,test]"
```
To run the tests, run the following command:
```console
$ pytest tests
```
If you want to add a new LLMClient you need to implement BaseLLMClient or BaseLLMAPIClient.

If you are adding a BaseLLMAPIClient you also need to add him in LLMAPIClientFactory.

You can add dependencies to your LLMClient in [pyproject.toml](pyproject.toml) also make sure you are adding a
matrix.flavor in [test.yml](.github%2Fworkflows%2Ftest.yml). 
