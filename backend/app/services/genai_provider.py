"""Shared Google GenAI client setup for model and embedding requests."""


def create_genai_client(api_key: str):
    """Create a client that does not inherit broken machine-level proxy settings.

    In particular, local development environments may inject an unreachable
    localhost proxy. The GenAI SDK's default HTTPX clients honor those settings,
    causing connection-refused errors before a request reaches Google.
    """
    from google import genai
    from google.genai import types

    http_options = types.HttpOptions(
        client_args={"trust_env": False},
        async_client_args={"trust_env": False},
    )
    return genai.Client(api_key=api_key, http_options=http_options)
