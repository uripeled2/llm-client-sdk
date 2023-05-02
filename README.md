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
For OpenAI support
```console
$ pip install llm-client[openai]
```
For AI21 support
```console
$ pip install llm-client[ai21]
```
For all current clients support
```console
$ pip install llm-client[all]
```

## Example
```python
import os
import openai
from llm_client import OpenAIClient

openai.api_key = os.environ["OPENAI_API_KEY"]
llm_client = OpenAIClient(client=openai, default_model="davinci")

llm_client.text_completion(prompt="This is indeed a test")
llm_client.text_completion(prompt="This is indeed a test", model="ada")
```
You can find more detailed example in the [examples/main.py](examples%2Fmain.py)

## Contributing

Contributions are welcome! Please check out the todos below, and feel free to open a pull request.

### Todo

- [ ] Add support for more LLMs
- [ ] Add support for more functions via LLMs (e.g. list models, edits, embeddings, etc.)
- [ ] Add support for local LLMs 
- [ ] Add contributing guidelines
- [ ] Create LLMClientFactory to more easily create LLMClient instances
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
If you want to add a new LLMClient you need to implement BaseLLMClient and adding the 
relevant dependencies in [pyproject.toml](pyproject.toml) also make sure you are adding a
matrix.flavor in [test.yml](.github%2Fworkflows%2Ftest.yml)