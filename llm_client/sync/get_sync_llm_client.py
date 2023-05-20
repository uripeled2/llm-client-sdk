import async_to_sync


def get_sync_llm_client(llm_client: object) -> async_to_sync.methods:
    return async_to_sync.methods(llm_client)
