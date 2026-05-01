"""Unified LLM client that routes to OpenAI, Anthropic, Together, or Groq."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

from tenacity import retry, stop_after_attempt, wait_exponential

_client_singleton: Optional["LLMClient"] = None

# Cost table: (prompt_cost_per_1M, completion_cost_per_1M)
# Keys are model name prefixes — versioned names like "gpt-4o-mini-2024-07-18" match "gpt-4o-mini".
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-haiku-3": (0.25, 1.25),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
}


def _lookup_cost(model: str) -> tuple[float, float]:
    """Exact match first, then longest-prefix match for versioned model names."""
    if model in _COST_TABLE:
        return _COST_TABLE[model]
    for key in sorted(_COST_TABLE, key=len, reverse=True):
        if model.startswith(key):
            return _COST_TABLE[key]
    return (0.0, 0.0)


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.calls += 1
            p_cost, c_cost = _lookup_cost(model)
            self.cost_usd += (prompt_tokens * p_cost + completion_tokens * c_cost) / 1_000_000
            log.debug(
                "cost model=%r prompt=%d completion=%d running_total=$%.6f",
                model, prompt_tokens, completion_tokens, self.cost_usd,
            )


class LLMClient:
    def __init__(self) -> None:
        self.usage = UsageStats()
        self._openai: object = None
        self._anthropic: object = None
        self._together: object = None
        self._groq: object = None
        self._init_lock = threading.Lock()

    def _get_openai(self) -> object:
        if self._openai is None:
            with self._init_lock:
                if self._openai is None:
                    from openai import OpenAI
                    self._openai = OpenAI(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        timeout=60.0,
                        max_retries=0,  # tenacity owns all retry logic
                    )
        return self._openai

    def _get_anthropic(self) -> object:
        if self._anthropic is None:
            with self._init_lock:
                if self._anthropic is None:
                    from anthropic import Anthropic
                    self._anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._anthropic

    def _get_together(self) -> object:
        if self._together is None:
            with self._init_lock:
                if self._together is None:
                    from openai import OpenAI
                    self._together = OpenAI(
                        api_key=os.environ.get("TOGETHER_API_KEY", ""),
                        base_url="https://api.together.xyz/v1",
                        max_retries=0,
                    )
        return self._together

    def _get_groq(self) -> object:
        if self._groq is None:
            with self._init_lock:
                if self._groq is None:
                    from openai import OpenAI
                    self._groq = OpenAI(
                        api_key=os.environ.get("GROQ_API_KEY", ""),
                        base_url="https://api.groq.com/openai/v1",
                        max_retries=0,
                    )
        return self._groq

    def _route(self, model: str) -> str:
        if model.startswith("claude-"):
            return "anthropic"
        if "/" in model:
            return "together"
        if "versatile" in model:
            return "groq"
        return "openai"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=90),
        reraise=True,
    )
    def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        provider = self._route(model)

        if provider == "anthropic":
            return self._complete_anthropic(model, system, user, temperature, max_tokens)
        else:
            client = {
                "openai": self._get_openai,
                "together": self._get_together,
                "groq": self._get_groq,
            }[provider]()
            return self._complete_openai_compat(client, model, system, user, temperature, max_tokens, json_mode)

    def _complete_anthropic(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        prompt_tokens = msg.usage.input_tokens
        completion_tokens = msg.usage.output_tokens
        self.usage.add(prompt_tokens, completion_tokens, model)
        return msg.content[0].text

    def _complete_openai_compat(
        self,
        client: object,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        usage = resp.usage
        if usage:
            self.usage.add(usage.prompt_tokens, usage.completion_tokens, model)
        return resp.choices[0].message.content or ""


def get_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton
