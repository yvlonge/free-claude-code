"""Parsing and validation helpers for provider/model target strings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightedTarget:
    """One provider/model target plus weight and canonical string."""

    provider_id: str
    model_name: str
    full_ref: str
    weight: int


def parse_single_target(
    value: str, *, supported_provider_ids: tuple[str, ...]
) -> tuple[str, str]:
    """Parse a single ``provider/model`` target and validate provider id."""
    raw = value.strip()
    if "/" not in raw:
        raise ValueError(
            "Model must be prefixed with provider type. "
            f"Valid providers: {', '.join(supported_provider_ids)}. "
            "Format: provider_type/model/name"
        )
    provider_id, model_name = raw.split("/", 1)
    if provider_id not in supported_provider_ids:
        supported = ", ".join(f"'{provider}'" for provider in supported_provider_ids)
        raise ValueError(f"Invalid provider: '{provider_id}'. Supported: {supported}")
    if not model_name.strip():
        raise ValueError("Model name must not be empty")
    return provider_id, model_name


def parse_weighted_target_pool(
    value: str,
    *,
    supported_provider_ids: tuple[str, ...],
) -> tuple[WeightedTarget, ...]:
    """Parse comma-separated weighted targets.

    Syntax:
    - ``provider/model``
    - ``provider/model@3``
    - ``provider/a@2, other/b@1``
    """
    entries = [part.strip() for part in value.split(",")]
    if not entries or any(not entry for entry in entries):
        raise ValueError("Model target pools must not contain empty entries")

    parsed: list[WeightedTarget] = []
    for entry in entries:
        target_ref = entry
        weight = 1
        if "@" in entry:
            maybe_ref, maybe_weight = entry.rsplit("@", 1)
            target_ref = maybe_ref.strip()
            weight_text = maybe_weight.strip()
            if not weight_text.isdigit():
                raise ValueError(
                    "Target weight must be a non-negative integer, got "
                    f"{weight_text!r} in {entry!r}"
                )
            weight = int(weight_text)
            if weight <= 0:
                raise ValueError(f"Target weight must be > 0 in {entry!r}")

        provider_id, model_name = parse_single_target(
            target_ref,
            supported_provider_ids=supported_provider_ids,
        )
        parsed.append(
            WeightedTarget(
                provider_id=provider_id,
                model_name=model_name,
                full_ref=f"{provider_id}/{model_name}",
                weight=weight,
            )
        )
    return tuple(parsed)
