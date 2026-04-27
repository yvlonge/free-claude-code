"""Model routing for Claude-compatible requests."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config.settings import Settings
from providers.registry import ProviderTargetPool

from .models.anthropic import MessagesRequest, TokenCountRequest


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    original_model: str
    provider_model_ref: str
    thinking_enabled: bool


@dataclass(frozen=True, slots=True)
class RoutedMessagesRequest:
    request: MessagesRequest
    resolved: ResolvedModel


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest
    resolved: ResolvedModel


class ModelRouter:
    """Resolve incoming Claude model names to configured provider model references."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def resolve(self, claude_model_name: str) -> ResolvedModel:
        provider_model_ref = self._settings.resolve_model(claude_model_name)
        targets = Settings.resolve_model_targets(provider_model_ref)
        first_target = targets[0]
        thinking_enabled = self._settings.resolve_thinking(claude_model_name)
        if first_target.model_name != claude_model_name:
            logger.debug(
                "MODEL MAPPING: '{}' -> '{}'",
                claude_model_name,
                first_target.model_name,
            )
        return ResolvedModel(
            original_model=claude_model_name,
            provider_model_ref=provider_model_ref,
            thinking_enabled=thinking_enabled,
        )

    @staticmethod
    def patch_request_for_target(
        request: MessagesRequest,
        target_pool: ProviderTargetPool,
        *,
        target_ref: str,
        original_model: str,
    ) -> MessagesRequest:
        """Build an attempt request with a concrete provider model target."""
        target = next(
            target for target in target_pool.targets if target.full_ref == target_ref
        )
        routed = request.model_copy(deep=True)
        routed.model = target.model_name
        routed.original_model = original_model
        routed.resolved_provider_model = target.full_ref
        return routed

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return a routed request context preserving unresolved pool reference."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        return RoutedMessagesRequest(request=routed, resolved=resolved)

    def resolve_token_count_request(
        self, request: TokenCountRequest
    ) -> RoutedTokenCountRequest:
        """Return an internal token-count request context."""
        resolved = self.resolve(request.model)
        first_target = Settings.resolve_model_targets(resolved.provider_model_ref)[0]
        routed = request.model_copy(
            update={
                "model": first_target.model_name,
                "original_model": request.model,
                "resolved_provider_model": first_target.full_ref,
            },
            deep=True,
        )
        return RoutedTokenCountRequest(request=routed, resolved=resolved)
