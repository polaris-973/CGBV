from __future__ import annotations

from pathlib import Path

import pytest

import cgbv.llm.openai_client as openai_client_mod
from cgbv.config.settings import LLMConfig, load_config
from cgbv.llm.factory import create_llm_client
from cgbv.llm.glm_client import GLMClient


class _DummyAsyncOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_load_config_defaults_glm_api_key_env(tmp_path: Path) -> None:
    config_path = tmp_path / "glm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  id: glm_test",
                '  description: "glm config test"',
                "dataset:",
                "  name: folio",
                "  split: validation",
                "llm:",
                "  provider: glm",
                "  model: glm-5",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.llm.api_key_env == "ZAI_API_KEY"


def test_create_llm_client_dispatches_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_client_mod, "AsyncOpenAI", _DummyAsyncOpenAI)

    client = create_llm_client(
        LLMConfig(
            provider="glm",
            model="glm-5",
            api_key_env="ZAI_API_KEY",
        )
    )

    assert isinstance(client, GLMClient)
    assert client._client.kwargs["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"


def test_glm_client_converts_zero_temperature_to_do_sample_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_client_mod, "AsyncOpenAI", _DummyAsyncOpenAI)

    client = GLMClient(
        LLMConfig(
            provider="glm",
            model="glm-5",
            api_key_env="ZAI_API_KEY",
            temperature=0.0,
            max_parallel_requests=4,
            min_request_interval=0.25,
        )
    )

    params = client._build_request_kwargs([{"role": "user", "content": "hi"}])

    assert "temperature" not in params
    assert params["extra_body"] == {"do_sample": False}
    assert client._request_semaphore is not None
    assert client._request_semaphore._value == 4
    assert client._min_request_interval == 0.25


def test_glm_client_preserves_positive_temperature_and_merges_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_client_mod, "AsyncOpenAI", _DummyAsyncOpenAI)

    client = GLMClient(
        LLMConfig(
            provider="glm",
            model="glm-4.7",
            api_key_env="ZAI_API_KEY",
            temperature=0.7,
            extra_body={"thinking": {"type": "enabled"}},
        )
    )

    params = client._build_request_kwargs([{"role": "user", "content": "hi"}])

    assert params["temperature"] == 0.7
    assert params["extra_body"] == {"thinking": {"type": "enabled"}}
