"""Application services for the Claude-compatible API."""

from __future__ import annotations

import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from inspect import isawaitable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from providers.base import BaseProvider
from providers.exceptions import (
    APIError,
    InvalidRequestError,
    OverloadedError,
    ProviderError,
    RateLimitError,
)
from providers.registry import ProviderTargetPool, transport_type_for_provider

from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]
TargetPoolGetter = Callable[[str], ProviderTargetPool]

_RETRYABLE_PROVIDER_ERRORS: tuple[type[ProviderError], ...] = (
    RateLimitError,
    OverloadedError,
)
_RETRYABLE_API_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


def _is_retryable_pre_output_failure(error: ProviderError) -> bool:
    if isinstance(error, _RETRYABLE_PROVIDER_ERRORS):
        return True
    if isinstance(error, APIError):
        return error.status_code in _RETRYABLE_API_STATUS_CODES
    return False


async def _stream_attempt(
    provider: BaseProvider,
    request: MessagesRequest,
    *,
    input_tokens: int,
    request_id: str,
    thinking_enabled: bool,
) -> tuple[str | None, AsyncIterator[str]]:
    """Open one upstream stream and eagerly fetch the first chunk if present."""
    body_or_awaitable = provider.stream_response(
        request,
        input_tokens=input_tokens,
        request_id=request_id,
        thinking_enabled=thinking_enabled,
    )

    if hasattr(body_or_awaitable, "__aiter__"):
        body = body_or_awaitable
    elif isawaitable(body_or_awaitable):
        resolved = await body_or_awaitable
        if not hasattr(resolved, "__aiter__"):
            raise TypeError("provider.stream_response() must return an async iterator")
        body = resolved
    else:
        raise TypeError("provider.stream_response() must return an async iterator")

    try:
        first_chunk = await anext(body)
    except StopAsyncIteration:
        return None, body
    return first_chunk, body


async def _stream_with_prefetched_chunk(
    first_chunk: str | None,
    body: AsyncIterator[str],
) -> AsyncIterator[str]:
    if first_chunk is not None:
        yield first_chunk
    async for chunk in body:
        yield chunk


async def _open_stream_with_failover(
    initial_request: MessagesRequest,
    *,
    initial_target_ref: str,
    routed_model,
    target_pool: ProviderTargetPool,
    provider_getter: ProviderGetter,
    model_router: ModelRouter,
    token_counter: TokenCounter,
    settings: Settings,
    request_id: str,
) -> tuple[str | None, AsyncIterator[str]]:
    """Return the first available upstream stream, failing over before output only."""
    attempted = {initial_target_ref}
    current_target_ref = initial_target_ref
    current_request = initial_request

    while True:
        current_provider_id = Settings.parse_provider_type(current_target_ref)
        provider = provider_getter(current_provider_id)
        current_tokens = token_counter(
            current_request.messages,
            current_request.system,
            current_request.tools,
        )

        try:
            return await _stream_attempt(
                provider,
                current_request,
                input_tokens=current_tokens,
                request_id=request_id,
                thinking_enabled=routed_model.thinking_enabled,
            )
        except ProviderError as error:
            if not _is_retryable_pre_output_failure(error):
                raise

            if len(target_pool.targets) == 1:
                raise

            target_pool.mark_unhealthy(
                current_target_ref,
                cooldown_seconds=settings.provider_target_cooldown_seconds,
            )

            while True:
                selected = target_pool.select_target()
                if selected is None:
                    retry_after = target_pool.retry_after_seconds() or 1
                    raise HTTPException(
                        status_code=503,
                        detail="No healthy upstream target is currently available.",
                        headers={"Retry-After": str(retry_after)},
                    ) from error

                next_target = selected.target
                if next_target.full_ref in attempted:
                    if len(attempted) >= len(target_pool.targets):
                        retry_after = target_pool.retry_after_seconds() or 1
                        raise HTTPException(
                            status_code=503,
                            detail="No healthy upstream target is currently available.",
                            headers={"Retry-After": str(retry_after)},
                        ) from error
                    continue

                attempted.add(next_target.full_ref)
                current_target_ref = next_target.full_ref
                current_request = model_router.patch_request_for_target(
                    initial_request,
                    target_pool,
                    target_ref=current_target_ref,
                    original_model=routed_model.original_model,
                )
                provider = provider_getter(next_target.provider_id)
                provider.preflight_stream(
                    current_request,
                    thinking_enabled=routed_model.thinking_enabled,
                )
                logger.info(
                    "API_RETRY: request_id={} model={} provider={}",
                    request_id,
                    current_request.model,
                    next_target.provider_id,
                )
                break


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        target_pool_getter: TargetPoolGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._target_pool_getter = target_pool_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def _validate_server_tool_compatibility(
        self,
        request: MessagesRequest,
        *,
        target_pool: ProviderTargetPool,
    ) -> None:
        for target in target_pool.targets:
            if transport_type_for_provider(target.provider_id) != "openai_chat":
                continue
            tool_err = openai_chat_upstream_server_tool_error(
                request,
                web_tools_enabled=self._settings.enable_web_server_tools,
            )
            if tool_err is not None:
                raise InvalidRequestError(tool_err)

    async def _stream_with_target_failover(
        self,
        request: MessagesRequest,
        *,
        routed_model,
    ) -> object:
        target_pool = self._target_pool_getter(routed_model.provider_model_ref)
        self._validate_server_tool_compatibility(request, target_pool=target_pool)

        selected = target_pool.select_target()
        if selected is None:
            retry_after = target_pool.retry_after_seconds() or 1
            raise HTTPException(
                status_code=503,
                detail="No healthy upstream target is currently available.",
                headers={"Retry-After": str(retry_after)},
            )

        target = selected.target
        routed_request = self._model_router.patch_request_for_target(
            request,
            target_pool,
            target_ref=target.full_ref,
            original_model=routed_model.original_model,
        )
        provider = self._provider_getter(target.provider_id)
        provider.preflight_stream(
            routed_request,
            thinking_enabled=routed_model.thinking_enabled,
        )

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        logger.info(
            "API_REQUEST: request_id={} model={} messages={} provider={}",
            request_id,
            routed_request.model,
            len(routed_request.messages),
            target.provider_id,
        )
        if self._settings.log_raw_api_payloads:
            logger.debug(
                "FULL_PAYLOAD [{}]: {}", request_id, routed_request.model_dump()
            )

        first_chunk, body = await _open_stream_with_failover(
            routed_request,
            initial_target_ref=target.full_ref,
            routed_model=routed_model,
            target_pool=target_pool,
            provider_getter=self._provider_getter,
            model_router=self._model_router,
            token_counter=self._token_counter,
            settings=self._settings,
            request_id=request_id,
        )

        return anthropic_sse_streaming_response(
            _stream_with_prefetched_chunk(
                first_chunk,
                body,
            )
        )

    async def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info("Optimization: Handling Anthropic web server tool")
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                return optimized
            logger.debug("No optimization matched, routing to provider")

            return await self._stream_with_target_failover(
                routed.request,
                routed_model=routed.resolved,
            )

        except ProviderError:
            raise
        except HTTPException:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info(
                    "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                    request_id,
                    routed.request.model,
                    len(routed.request.messages),
                    tokens,
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
