# LLM-Client-SDK
LLM-Client-SDK is an SDK for communicating with generative AI large language models.

It is design to enable easily integration with different LLM and to easily switch between them

## Requirements

Python 3.9+

## Installation
```console
$ pip install llm-client
```
### Optional Dependencies
For all current clients support
```console
$ pip install llm-client[all]
```
For only current api clients support
```console
$ pip install llm-client[api]
```
For only OpenAI support
```console
$ pip install llm-client[openai]
```
For only AI21 support
```console
$ pip install llm-client[ai21]
```
For only local client support
```console
$ pip install llm-client[local]
```

## Examples

### OpenAI
```python
import os
from aiohttp import ClientSession
from llm_client import get_llm_client, LLMAPIClientConfigWithClientType, LLMAPIClientType, LLMAPIClientConfig

async def main():
    async with ClientSession() as session:
        llm_client = get_llm_client(LLMAPIClientConfigWithClientType(LLMAPIClientType.OPEN_AI, 
                                                    LLMAPIClientConfig(os.environ["OPENAI_API_KEY"], session,
                                                                        default_model="ada")))

        await llm_client.text_completion(prompt="This is indeed a test")
        await llm_client.text_completion(prompt="This is indeed a test", model="text-davinci-003")
```

### Local
```python
import os
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from llm_client import get_llm_client, LocalClientConfig

async def main():
    try:
        model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
    except ValueError:
        model = AutoModelForSeq2SeqLM.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_NAME_OR_PATH"])
    llm_client = get_llm_client(LocalClientConfig(model, tokenizer, os.environ["TENSORS_TYPE"], os.environ["DEVICE"]))

    await llm_client.text_completion(prompt="This is indeed a test")
    await llm_client.text_completion(prompt="This is indeed a test", max_length=50)
```
You can find more detailed example in [examples](examples)

## Contributing

Contributions are welcome! Please check out the todos below, and feel free to open issue or a pull request.

### Todo

- [ ] Add support for more LLMs
- [ ] Add support for more functions via LLMs (e.g. embeddings, list models, edits, etc.)
- [ ] Add contributing guidelines
- [ ] Create an easy way to run multiple LLMs in parallel with the same prompts

### Development
To install the package in development mode, run the following command:
```console
$ pip install -e ".[all,test]"
```
To run the tests, run the following command:
```console
$ pytest tests
```
If you want to add a new LLMClient you need to implement BaseLLMClient or BaseLLMAPIClient and adding the 
relevant dependencies in [pyproject.toml](pyproject.toml) also make sure you are adding a
matrix.flavor in [test.yml](.github%2Fworkflows%2Ftest.yml)