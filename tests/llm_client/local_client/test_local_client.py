from unittest.mock import call

import pytest

from llm_client.llm_client.local_client import LocalClient, LocalClientConfig


@pytest.mark.asyncio
async def test_text_completion__sanity(local_client, mock_model, mock_tokenizer, tensors_type, device):
    mock_tokenizer.encode.return_value.to.return_value = [1, 2, 3]
    mock_model.generate.return_value = [1]
    mock_tokenizer.decode.return_value = "first completion"

    actual = await local_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["first completion"]
    mock_tokenizer.encode.assert_called_once_with("These are a few of my favorite", return_tensors=tensors_type)
    mock_tokenizer.encode.return_value.to.assert_called_once_with(device)
    mock_model.generate.assert_called_once_with([1, 2, 3])
    mock_tokenizer.decode.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_text_completion__return_multiple_completions(local_client, mock_model, mock_tokenizer, tensors_type,
                                                            device):
    mock_tokenizer.encode.return_value.to.return_value = [1, 2, 3]
    mock_model.generate.return_value = [2, 3, 4]
    mock_tokenizer.decode.side_effect = ["first completion", "second completion", "third completion"]

    actual = await local_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["first completion", "second completion", "third completion"]
    mock_tokenizer.encode.assert_called_once_with("These are a few of my favorite", return_tensors=tensors_type)
    mock_tokenizer.encode.return_value.to.assert_called_once_with(device)
    mock_model.generate.assert_called_once_with([1, 2, 3])
    assert mock_tokenizer.decode.call_args_list == [call(2), call(3), call(4)]


@pytest.mark.asyncio
async def test_text_completion__with_kwargs(local_client, mock_model, mock_tokenizer, tensors_type, device):
    mock_tokenizer.encode.return_value.to.return_value = [1, 2, 3]
    mock_model.generate.return_value = [1]
    mock_tokenizer.decode.return_value = "first completion"

    actual = await local_client.text_completion(prompt="These are a few of my favorite", max_length=100)

    assert actual == ["first completion"]
    mock_tokenizer.encode.assert_called_once_with("These are a few of my favorite", return_tensors=tensors_type)
    mock_tokenizer.encode.return_value.to.assert_called_once_with(device)
    mock_model.generate.assert_called_once_with([1, 2, 3], max_length=100)
    mock_tokenizer.decode.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_text_completion__with_encode_kwargs(mock_model, mock_tokenizer, tensors_type, device):
    mock_tokenizer.encode.return_value.to.return_value = [1, 2, 3]
    mock_model.generate.return_value = [1]
    mock_tokenizer.decode.return_value = "first completion"
    encode_kwargs = {"add_special_tokens": False}
    local_client = LocalClient(LocalClientConfig(mock_model, mock_tokenizer, tensors_type, device, encode_kwargs))

    actual = await local_client.text_completion(prompt="These are a few of my favorite")

    assert actual == ["first completion"]
    mock_tokenizer.encode.assert_called_once_with("These are a few of my favorite", return_tensors=tensors_type,
                                                  add_special_tokens=False)
    mock_tokenizer.encode.return_value.to.assert_called_once_with(device)
    mock_model.generate.assert_called_once_with([1, 2, 3])
    mock_tokenizer.decode.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_get_tokens_count__sanity(local_client, mock_tokenizer, tensors_type, device):
    mock_tokenizer.encode.return_value.to.return_value = [1, 2, 3]

    actual = await local_client.get_tokens_count(text="This is a test")

    assert actual == 3
    mock_tokenizer.encode.assert_called_once_with("This is a test", return_tensors=tensors_type)
    mock_tokenizer.encode.return_value.to.assert_called_once_with(device)


@pytest.mark.asyncio
async def test_get_tokens_count__with_kwargs(mock_model, mock_tokenizer, tensors_type, device):
    mock_tokenizer.encode.return_value.to.return_value = [1, 2, 3]
    encode_kwargs = {"add_special_tokens": False}
    local_client = LocalClient(LocalClientConfig(mock_model, mock_tokenizer, tensors_type, device, encode_kwargs))

    actual = await local_client.get_tokens_count(text="This is a test")

    assert actual == 3
    mock_tokenizer.encode.assert_called_once_with("This is a test", return_tensors=tensors_type,
                                                  add_special_tokens=False)
    mock_tokenizer.encode.return_value.to.assert_called_once_with(device)
