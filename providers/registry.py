"""Provider descriptors, factory, and runtime registry."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

from config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderDescriptor,
)
from config.settings import Settings
from core.scheduling import WeightedTargetScheduler, WeightedTargetSlot
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import AuthenticationError, UnknownProviderTypeError

ProviderFactory = Callable[[ProviderConfig, Settings], BaseProvider]

# Backwards-compatible name for the catalog (single source: ``config.provider_catalog``).
PROVIDER_DESCRIPTORS: dict[str, ProviderDescriptor] = PROVIDER_CATALOG


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    """One concrete provider/model target used for one upstream attempt."""

    provider_id: str
    model_name: str
    full_ref: str
    weight: int


@dataclass(frozen=True, slots=True)
class TargetSelection:
    """Chosen target plus pool retry metadata."""

    target: ProviderTarget
    retry_after_seconds: int


class ProviderTargetPool:
    """Weighted target pool with stateful cooldown scheduling."""

    def __init__(self, targets: tuple[ProviderTarget, ...]) -> None:
        if not targets:
            raise ValueError("ProviderTargetPool requires at least one target")
        self._targets = targets
        self._target_by_ref = {target.full_ref: target for target in targets}
        self._scheduler = WeightedTargetScheduler(
            tuple(
                WeightedTargetSlot(target_ref=target.full_ref, weight=target.weight)
                for target in targets
            )
        )

    @property
    def targets(self) -> tuple[ProviderTarget, ...]:
        return self._targets

    def first_target(self) -> ProviderTarget:
        return self._targets[0]

    def select_target(self) -> TargetSelection | None:
        selected = self._scheduler.next_target()
        if selected is None:
            return None
        target = self._target_by_ref[selected.target_ref]
        return TargetSelection(
            target=target,
            retry_after_seconds=selected.retry_after_seconds,
        )

    def mark_unhealthy(self, target_ref: str, *, cooldown_seconds: float) -> None:
        self._scheduler.mark_unhealthy(target_ref, cooldown_seconds=cooldown_seconds)

    def retry_after_seconds(self) -> int | None:
        snapshot = self._scheduler.cooldown_snapshot()
        if snapshot is None:
            return None
        return snapshot.remaining_seconds


def _create_nvidia_nim(config: ProviderConfig, settings: Settings) -> BaseProvider:
    from providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(config, nim_settings=settings.nim)


def _create_open_router(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config)


def _create_deepseek(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config)


def _create_lmstudio(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config)


def _create_llamacpp(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.llamacpp import LlamaCppProvider

    return LlamaCppProvider(config)


def _create_ollama(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.ollama import OllamaProvider

    return OllamaProvider(config)


def _create_local_api(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.local_api import LocalAPIProvider

    return LocalAPIProvider(config)


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "llamacpp": _create_llamacpp,
    "ollama": _create_ollama,
    "local_api": _create_local_api,
}

if set(PROVIDER_DESCRIPTORS) != set(SUPPORTED_PROVIDER_IDS) or set(
    PROVIDER_FACTORIES
) != set(SUPPORTED_PROVIDER_IDS):
    raise AssertionError(
        "PROVIDER_DESCRIPTORS, PROVIDER_FACTORIES, and SUPPORTED_PROVIDER_IDS are out of sync: "
        f"descriptors={set(PROVIDER_DESCRIPTORS)!r} factories={set(PROVIDER_FACTORIES)!r} "
        f"ids={set(SUPPORTED_PROVIDER_IDS)!r}"
    )


def _string_attr(settings: Settings, attr_name: str | None, default: str = "") -> str:
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def _credential_for(descriptor: ProviderDescriptor, settings: Settings) -> str:
    if descriptor.credential_attr:
        credential = _string_attr(settings, descriptor.credential_attr)
        if credential.strip():
            return credential
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    return ""


def _require_credential(descriptor: ProviderDescriptor, credential: str) -> None:
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    message = f"{descriptor.credential_env} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise AuthenticationError(message)


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings
) -> ProviderConfig:
    credential = _credential_for(descriptor, settings)
    _require_credential(descriptor, credential)
    base_url = _string_attr(
        settings, descriptor.base_url_attr, descriptor.default_base_url or ""
    )
    proxy = _string_attr(settings, descriptor.proxy_attr)
    return ProviderConfig(
        api_key=credential,
        base_url=base_url or descriptor.default_base_url,
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        enable_thinking=settings.enable_model_thinking,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )


def create_provider(provider_id: str, settings: Settings) -> BaseProvider:
    descriptor = PROVIDER_DESCRIPTORS.get(provider_id)
    if descriptor is None:
        supported = "', '".join(PROVIDER_DESCRIPTORS)
        raise UnknownProviderTypeError(
            f"Unknown provider_type: '{provider_id}'. Supported: '{supported}'"
        )

    config = build_provider_config(descriptor, settings)
    factory = PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        raise AssertionError(f"Unhandled provider descriptor: {provider_id}")
    return factory(config, settings)


def transport_type_for_provider(provider_id: str) -> str:
    descriptor = PROVIDER_DESCRIPTORS.get(provider_id)
    if descriptor is None:
        supported = "', '".join(PROVIDER_DESCRIPTORS)
        raise UnknownProviderTypeError(
            f"Unknown provider_type: '{provider_id}'. Supported: '{supported}'"
        )
    return descriptor.transport_type


def _canonical_pool_key(targets: tuple[ProviderTarget, ...]) -> str:
    entries: list[str] = []
    for target in targets:
        if target.weight == 1:
            entries.append(target.full_ref)
        else:
            entries.append(f"{target.full_ref}@{target.weight}")
    return ",".join(entries)


def _pool_targets_from_model_ref(model_ref: str) -> tuple[ProviderTarget, ...]:
    weighted = Settings.resolve_model_targets(model_ref)
    return tuple(
        ProviderTarget(
            provider_id=target.provider_id,
            model_name=target.model_name,
            full_ref=target.full_ref,
            weight=target.weight,
        )
        for target in weighted
    )


class ProviderRegistry:
    """Cache provider instances and parsed model target pools."""

    def __init__(
        self,
        providers: MutableMapping[str, BaseProvider] | None = None,
        target_pools: MutableMapping[str, ProviderTargetPool] | None = None,
    ):
        self._providers = providers if providers is not None else {}
        self._target_pools = target_pools if target_pools is not None else {}

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already in the cache."""
        return provider_id in self._providers

    def get(self, provider_id: str, settings: Settings) -> BaseProvider:
        if provider_id not in self._providers:
            self._providers[provider_id] = create_provider(provider_id, settings)
        return self._providers[provider_id]

    def get_target_pool(self, model_ref: str) -> ProviderTargetPool:
        targets = _pool_targets_from_model_ref(model_ref)
        key = _canonical_pool_key(targets)
        pool = self._target_pools.get(key)
        if pool is None:
            pool = ProviderTargetPool(targets)
            self._target_pools[key] = pool
        return pool

    async def cleanup(self) -> None:
        """Call ``cleanup`` on every cached provider, then clear provider and pool state.

        Attempts all providers even if one fails. A single failure is re-raised
        as-is; multiple failures are wrapped in :exc:`ExceptionGroup`.
        """
        items = list(self._providers.items())
        errors: list[Exception] = []
        try:
            for _pid, provider in items:
                try:
                    await provider.cleanup()
                except Exception as e:
                    errors.append(e)
        finally:
            self._providers.clear()
            self._target_pools.clear()
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            msg = "One or more provider cleanups failed"
            raise ExceptionGroup(msg, errors)
