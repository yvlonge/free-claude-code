from __future__ import annotations

import pytest

from config.model_targets import parse_single_target, parse_weighted_target_pool

_SUPPORTED = (
    "nvidia_nim",
    "open_router",
    "deepseek",
    "lmstudio",
    "llamacpp",
    "ollama",
    "local_api",
)


def test_parse_single_target_valid() -> None:
    provider_id, model_name = parse_single_target(
        "local_api/my-model",
        supported_provider_ids=_SUPPORTED,
    )

    assert provider_id == "local_api"
    assert model_name == "my-model"


def test_parse_single_target_rejects_missing_provider_prefix() -> None:
    with pytest.raises(ValueError, match="provider type"):
        parse_single_target("plain-model", supported_provider_ids=_SUPPORTED)


def test_parse_weighted_target_pool_parses_weights() -> None:
    targets = parse_weighted_target_pool(
        "local_api/foo@3, open_router/bar@1",
        supported_provider_ids=_SUPPORTED,
    )

    assert [target.full_ref for target in targets] == [
        "local_api/foo",
        "open_router/bar",
    ]
    assert [target.weight for target in targets] == [3, 1]


def test_parse_weighted_target_pool_defaults_weight_to_one() -> None:
    targets = parse_weighted_target_pool(
        "deepseek/chat",
        supported_provider_ids=_SUPPORTED,
    )

    assert targets[0].weight == 1


def test_parse_weighted_target_pool_rejects_zero_weight() -> None:
    with pytest.raises(ValueError, match="weight must be > 0"):
        parse_weighted_target_pool(
            "local_api/foo@0",
            supported_provider_ids=_SUPPORTED,
        )
