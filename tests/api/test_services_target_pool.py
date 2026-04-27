from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.base import BaseProvider
from providers.exceptions import OverloadedError, RateLimitError
from providers.registry import ProviderTarget, ProviderTargetPool


class FailingProvider(BaseProvider):
    def __init__(self, exc: Exception) -> None:
        from providers.base import ProviderConfig

        super().__init__(ProviderConfig(api_key="k"))
        self._exc = exc

    def preflight_stream(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    async def cleanup(self) -> None:
        return None

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        del request, input_tokens, request_id, thinking_enabled
        raise self._exc
        if False:
            yield ""


class RecordingProvider(BaseProvider):
    def __init__(self) -> None:
        from providers.base import ProviderConfig

        super().__init__(ProviderConfig(api_key="k"))
        self.seen_models: list[str] = []

    async def cleanup(self) -> None:
        return None

    def preflight_stream(self, request, **_kwargs) -> None:
        self.seen_models.append(request.model)

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, thinking_enabled
        self.seen_models.append(request.model)
        yield "event: message_start\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"


async def _stream_text(response) -> str:
    iterator = response.body_iterator
    if hasattr(iterator, "__aiter__"):
        parts = [str(part) async for part in iterator]
        return "".join(parts)
    if hasattr(iterator, "__iter__"):
        return "".join(str(part) for part in iterator)
    return ""


def _request() -> MessagesRequest:
    return MessagesRequest(
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        messages=[Message(role="user", content="hello")],
    )


def test_service_failover_marks_retryable_target_unhealthy() -> None:
    settings = Settings()
    settings.model = "nvidia_nim/fallback"

    pool = ProviderTargetPool(
        (
            ProviderTarget(
                provider_id="nvidia_nim",
                model_name="m1",
                full_ref="nvidia_nim/m1",
                weight=1,
            ),
            ProviderTarget(
                provider_id="deepseek",
                model_name="m2",
                full_ref="deepseek/m2",
                weight=1,
            ),
        )
    )

    providers = {
        "nvidia_nim": FailingProvider(RateLimitError("rate")),
        "deepseek": RecordingProvider(),
    }

    service = ClaudeProxyService(
        settings,
        provider_getter=lambda provider_id: providers[provider_id],
        target_pool_getter=lambda _model_ref: pool,
    )

    async def _create_and_collect() -> tuple[object, str]:
        response = await service.create_message(_request())
        return response, await _stream_text(response)

    response, stream_text = asyncio.run(_create_and_collect())

    assert response is not None
    assert providers["deepseek"].seen_models
    assert "message_stop" in stream_text


def test_service_returns_503_when_all_targets_unhealthy() -> None:
    settings = Settings()
    settings.model = "nvidia_nim/fallback"
    settings.provider_target_cooldown_seconds = 60

    pool = ProviderTargetPool(
        (
            ProviderTarget(
                provider_id="nvidia_nim",
                model_name="m1",
                full_ref="nvidia_nim/m1",
                weight=1,
            ),
            ProviderTarget(
                provider_id="deepseek",
                model_name="m2",
                full_ref="deepseek/m2",
                weight=1,
            ),
        )
    )

    providers = {
        "nvidia_nim": FailingProvider(OverloadedError("boom")),
        "deepseek": FailingProvider(RateLimitError("rate")),
    }

    service = ClaudeProxyService(
        settings,
        provider_getter=lambda provider_id: providers[provider_id],
        target_pool_getter=lambda _model_ref: pool,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(service.create_message(_request()))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "No healthy upstream target is currently available."
    assert exc_info.value.headers == {"Retry-After": "60"}


def test_service_does_not_failover_non_retryable_errors() -> None:
    settings = Settings()
    settings.model = "nvidia_nim/fallback"

    pool = ProviderTargetPool(
        (
            ProviderTarget(
                provider_id="nvidia_nim",
                model_name="m1",
                full_ref="nvidia_nim/m1",
                weight=1,
            ),
            ProviderTarget(
                provider_id="deepseek",
                model_name="m2",
                full_ref="deepseek/m2",
                weight=1,
            ),
        )
    )

    failing_provider = FailingProvider(Exception("unexpected"))
    fallback_provider = MagicMock()

    service = ClaudeProxyService(
        settings,
        provider_getter=lambda provider_id: (
            failing_provider if provider_id == "nvidia_nim" else fallback_provider
        ),
        target_pool_getter=lambda _model_ref: pool,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(service.create_message(_request()))

    assert exc_info.value.status_code == 500
    fallback_provider.preflight_stream.assert_not_called()


def test_service_does_not_failover_invalid_request_error() -> None:
    from providers.exceptions import InvalidRequestError

    settings = Settings()
    settings.model = "nvidia_nim/fallback"

    pool = ProviderTargetPool(
        (
            ProviderTarget(
                provider_id="nvidia_nim",
                model_name="m1",
                full_ref="nvidia_nim/m1",
                weight=1,
            ),
            ProviderTarget(
                provider_id="deepseek",
                model_name="m2",
                full_ref="deepseek/m2",
                weight=1,
            ),
        )
    )

    failing_provider = FailingProvider(InvalidRequestError("bad request"))
    fallback_provider = MagicMock()

    service = ClaudeProxyService(
        settings,
        provider_getter=lambda provider_id: (
            failing_provider if provider_id == "nvidia_nim" else fallback_provider
        ),
        target_pool_getter=lambda _model_ref: pool,
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(service.create_message(_request()))

    fallback_provider.preflight_stream.assert_not_called()
