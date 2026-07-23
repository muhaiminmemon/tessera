"""Tests for the LLM client router, cost table, and provider key checks."""
from __future__ import annotations

import pytest

from tessera import _check_api_key_for_model
from tessera.core.exceptions import ConfigurationError
from tessera.core.llm_client import LLMClient, UsageStats, _lookup_cost


class TestRoute:
    @pytest.fixture()
    def client(self) -> LLMClient:
        return LLMClient()

    def test_openai_models(self, client: LLMClient) -> None:
        assert client._route("gpt-4o-mini") == "openai"
        assert client._route("gpt-4.1-nano") == "openai"

    def test_anthropic_models(self, client: LLMClient) -> None:
        assert client._route("claude-3-haiku-20240307") == "anthropic"
        assert client._route("claude-3-5-sonnet-20241022") == "anthropic"

    def test_together_models(self, client: LLMClient) -> None:
        assert client._route("meta-llama/Llama-3.3-70B-Instruct-Turbo") == "together"

    def test_groq_models(self, client: LLMClient) -> None:
        # Groq serves both "instant" and "versatile" Llama variants
        assert client._route("llama-3.1-8b-instant") == "groq"
        assert client._route("llama-3.3-70b-versatile") == "groq"


class TestCostTable:
    def test_exact_match(self) -> None:
        assert _lookup_cost("gpt-4o-mini") == (0.15, 0.60)

    def test_versioned_name_prefix_match(self) -> None:
        # Longest prefix wins: versioned gpt-4o-mini must not match plain gpt-4o
        assert _lookup_cost("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
        assert _lookup_cost("gpt-4o-2024-08-06") == (5.00, 15.00)

    def test_gpt41_family_present(self) -> None:
        assert _lookup_cost("gpt-4.1-mini") == (0.40, 1.60)
        assert _lookup_cost("gpt-4.1-nano") == (0.10, 0.40)
        assert _lookup_cost("gpt-4.1") == (2.00, 8.00)

    def test_groq_model_present(self) -> None:
        assert _lookup_cost("llama-3.1-8b-instant") == (0.05, 0.08)

    def test_unknown_model_costs_zero(self) -> None:
        assert _lookup_cost("some-future-model") == (0.0, 0.0)


class TestUsageStats:
    def test_accumulates_cost(self) -> None:
        stats = UsageStats()
        stats.add(1_000_000, 1_000_000, "gpt-4o-mini")
        assert stats.prompt_tokens == 1_000_000
        assert stats.completion_tokens == 1_000_000
        assert stats.calls == 1
        assert stats.cost_usd == pytest.approx(0.75)


class TestApiKeyCheck:
    def test_missing_openai_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            _check_api_key_for_model("gpt-4o-mini")

    def test_anthropic_model_needs_anthropic_key_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        # Must not require OPENAI_API_KEY for a claude model
        _check_api_key_for_model("claude-3-haiku-20240307")

    def test_groq_model_needs_groq_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            _check_api_key_for_model("llama-3.1-8b-instant")
