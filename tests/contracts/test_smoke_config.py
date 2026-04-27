from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smoke.lib.config import DEFAULT_TARGETS, TARGET_REQUIRED_ENV, SmokeConfig


def _settings(**overrides):
    values = {
        "model": "ollama/llama3.1",
        "model_opus": None,
        "model_sonnet": None,
        "model_haiku": None,
        "nvidia_nim_api_key": "",
        "open_router_api_key": "",
        "deepseek_api_key": "",
        "lm_studio_base_url": "",
        "llamacpp_base_url": "",
        "ollama_base_url": "http://localhost:11434",
        "local_api_base_url": "http://127.0.0.1:4000/v1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _smoke_config(**overrides) -> SmokeConfig:
    values = {
        "root": Path("."),
        "results_dir": Path(".smoke-results"),
        "live": False,
        "interactive": False,
        "targets": DEFAULT_TARGETS,
        "provider_matrix": frozenset(),
        "timeout_s": 45.0,
        "prompt": "Reply with exactly: FCC_SMOKE_PONG",
        "claude_bin": "claude",
        "worker_id": "main",
        "settings": _settings(),
    }
    values.update(overrides)
    return SmokeConfig(**values)


def test_ollama_is_default_smoke_target() -> None:
    assert "ollama" in DEFAULT_TARGETS
    assert "ollama" in TARGET_REQUIRED_ENV


def test_local_api_is_default_smoke_target() -> None:
    assert "local_api" in DEFAULT_TARGETS
    assert "local_api" in TARGET_REQUIRED_ENV


def test_ollama_provider_configuration_uses_base_url() -> None:
    config = _smoke_config()

    assert config.has_provider_configuration("ollama")
    assert config.provider_models()[0].full_model == "ollama/llama3.1"


def test_local_api_provider_configuration_uses_base_url() -> None:
    config = _smoke_config(settings=_settings(model="local_api/local-model"))

    assert config.has_provider_configuration("local_api")
    assert config.provider_models()[0].full_model == "local_api/local-model"


def test_ollama_provider_matrix_filters_models() -> None:
    config = _smoke_config(provider_matrix=frozenset({"ollama"}))

    assert [model.provider for model in config.provider_models()] == ["ollama"]


def test_provider_models_expands_weighted_pool_entries() -> None:
    config = _smoke_config(
        settings=_settings(model="ollama/llama3.1@2, local_api/local-model@1")
    )

    assert [model.full_model for model in config.provider_models()] == [
        "ollama/llama3.1",
        "local_api/local-model",
    ]
