"""Tests for local_api provider."""

from unittest.mock import AsyncMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from providers.base import ProviderConfig
from providers.local_api import LOCAL_API_DEFAULT_BASE, LocalAPIProvider


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


def test_init_uses_local_api_defaults() -> None:
    config = ProviderConfig(api_key="", base_url=LOCAL_API_DEFAULT_BASE)
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = LocalAPIProvider(config)

    assert provider._base_url == LOCAL_API_DEFAULT_BASE
    mock_openai.assert_called_once()


def test_build_request_body_maps_messages_request() -> None:
    provider = LocalAPIProvider(
        ProviderConfig(api_key="", base_url=LOCAL_API_DEFAULT_BASE)
    )
    request = MessagesRequest(
        model="local-model",
        max_tokens=50,
        messages=[Message(role="user", content="hello")],
    )

    body = provider._build_request_body(request)

    assert body["model"] == "local-model"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello"


def test_registry_builds_local_api_provider() -> None:
    from config.settings import Settings
    from providers.registry import create_provider

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = create_provider("local_api", Settings())

    assert isinstance(provider, LocalAPIProvider)
