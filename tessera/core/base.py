"""Abstract base class that all task implementations inherit from."""
from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tessera.core.llm_client import get_client
from tessera.core.models import (
    Example,
    GenerationResult,
    Persona,
    TaskSpec,
    TaskType,
    Taxonomy,
)


class TaskTemplate(ABC):
    """Defines the contract every task type must fulfil."""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_taxonomy(self, spec: TaskSpec) -> Taxonomy:
        raise NotImplementedError

    @abstractmethod
    def generate_example(
        self, node: Any, persona: Persona, spec: TaskSpec
    ) -> Example:
        raise NotImplementedError

    @abstractmethod
    def critique_example(self, example: Example, spec: TaskSpec) -> Example:
        """Score example and set passed_critique; return modified example."""
        raise NotImplementedError

    @abstractmethod
    def deduplicate(self, examples: list[Example]) -> list[Example]:
        raise NotImplementedError

    @abstractmethod
    def format_for_finetuning(
        self, examples: list[Example], fmt: str = "jsonl"
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def validate_downstream(
        self, train: list[Example], test: list[Example]
    ) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete pipeline orchestration
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        spec: TaskSpec,
        personas: list[Persona],
        n_examples: int,
        critique_threshold: float = 6.0,
        max_attempts_multiplier: float = 2.5,
    ) -> GenerationResult:
        """End-to-end pipeline: taxonomy → generate → critique → dedup → trim."""
        taxonomy = self.build_taxonomy(spec)

        target_raw = int(n_examples * max_attempts_multiplier)
        nodes_sample = self._sample_nodes_balanced(taxonomy, target_raw)

        cost_before = get_client().usage.cost_usd
        max_workers = int(os.environ.get("TESSERA_MAX_CONCURRENT", "10"))

        # --- Parallel generation ---
        def _gen_worker(node: Any) -> Example | None:
            persona = random.choice(personas)
            try:
                return self.generate_example(node, persona, spec)
            except Exception as exc:
                print(f"[tessera] generation failed for node {node.id}: {exc}")
                return None

        raw_examples: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_gen_worker, node) for node in nodes_sample]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    raw_examples.append(result)
                if i % 10 == 0:
                    print(f"[tessera] generating {i}/{target_raw}...")

        total_generated = len(raw_examples)

        # --- Parallel critique via score_batch (routes through task's critique_example
        #     to preserve any task-specific post-processing) ---
        def _critique_worker(ex: Example) -> Example | None:
            try:
                return self.critique_example(ex, spec)
            except Exception as exc:
                print(f"[tessera] critique failed for example {ex.id}: {exc}")
                return None

        scored: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_c = {executor.submit(_critique_worker, ex): ex for ex in raw_examples}
            for future in as_completed(futures_c):
                result = future.result()
                if result is not None:
                    scored.append(result)

        passed = [ex for ex in scored if ex.passed_critique]
        total_after_critique = len(passed)

        if scored and total_after_critique == len(scored):
            print(
                "[tessera] warning: 100% critique pass rate — "
                "consider raising threshold to 7.0"
            )

        # Dedup
        deduped = self.deduplicate(passed)
        total_after_dedup = len(deduped)

        # Trim to requested size
        final = deduped[:n_examples]

        cost_usd = get_client().usage.cost_usd - cost_before

        return GenerationResult(
            task_type=self._task_type(),
            spec=spec.model_dump(),
            examples=final,
            total_generated=total_generated,
            total_after_critique=total_after_critique,
            total_after_dedup=total_after_dedup,
            cost_usd=cost_usd,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _task_type(self) -> TaskType:
        raise NotImplementedError("subclass must override _task_type()")

    @staticmethod
    def _sample_nodes_balanced(taxonomy: Taxonomy, n: int) -> list[Any]:
        """Round-robin over labels to produce a balanced node sample."""
        labels = list({node.target_label for node in taxonomy.nodes})
        if not labels:
            return []

        buckets: dict[str, list[Any]] = {lbl: taxonomy.nodes_for_label(lbl) for lbl in labels}
        result: list[Any] = []
        label_cycle = list(labels)
        idx = 0

        while len(result) < n:
            label = label_cycle[idx % len(label_cycle)]
            bucket = buckets[label]
            if bucket:
                result.append(random.choice(bucket))
            idx += 1
            if idx > n * 10:
                break

        random.shuffle(result)
        return result
