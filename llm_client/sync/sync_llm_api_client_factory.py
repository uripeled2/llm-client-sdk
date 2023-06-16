import inspect
from functools import wraps
from typing import Callable, Any

import async_to_sync
from aiohttp import ClientSession

from llm_client import LLMAPIClientConfig
from llm_client.llm_api_client.llm_api_client_factory import LLMAPIClientType, get_llm_api_client_class


def _create_new_session(f: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(f)
    async def func(self, *args, **kwargs):
        self._session = ClientSession()
        try:
            response = await f(self, *args, **kwargs)
        except Exception as e:
            await self._session.close()
            raise e
        await self._session.close()
        return response

    return func


def _decorate_all_methods_in_class(decorators):
    def apply_decorator(cls: Any) -> Any:
        for k, f in cls.__dict__.items():
            if inspect.isfunction(f) and not k.startswith("_"):
                for decorator in decorators:
                    setattr(cls, k, decorator(cls.__dict__[k]))
        return cls

    return apply_decorator


def init_sync_llm_api_client(llm_api_client_type: LLMAPIClientType, **config_kwargs) -> "Sync BaseLLMAPIClient":
    llm_api_client_class = get_llm_api_client_class(llm_api_client_type)
    return async_to_sync.methods(_decorate_all_methods_in_class([_create_new_session])(llm_api_client_class)(
        LLMAPIClientConfig(session=None, **config_kwargs)))
