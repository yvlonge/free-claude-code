"""Request builder for local_api provider."""

from typing import Any

from core.anthropic import build_base_request_body
from core.anthropic.conversion import OpenAIConversionError, ReasoningReplayMode
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for local_api."""
    try:
        return build_base_request_body(
            request_data,
            reasoning_replay=(
                ReasoningReplayMode.THINK_TAGS
                if thinking_enabled
                else ReasoningReplayMode.DISABLED
            ),
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc
