"""local_api provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import LOCAL_API_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class LocalAPIProvider(OpenAIChatTransport):
    """OpenAI-compatible local API provider adapter."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LOCAL_API",
            base_url=config.base_url or LOCAL_API_DEFAULT_BASE,
            api_key=config.api_key or "local-api",
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
