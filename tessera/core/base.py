"""Abstract base class that all task implementations inherit from."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

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

        raw_examples: list[Example] = []
        for node in nodes_sample:
            persona = random.choice(personas)
            try:
                ex = self.generate_example(node, persona, spec)
                raw_examples.append(ex)
            except Exception as exc:
                print(f"[tessera] generation failed for node {node.id}: {exc}")

        total_generated = len(raw_examples)

        # Critique filter
        scored: list[Example] = []
        for ex in raw_examples:
            try:
                ex = self.critique_example(ex, spec)
                scored.append(ex)
            except Exception as exc:
                print(f"[tessera] critique failed for example {ex.id}: {exc}")

        passed = [ex for ex in scored if ex.passed_critique]
        total_after_critique = len(passed)

        # Dedup
        deduped = self.deduplicate(passed)
        total_after_dedup = len(deduped)

        # Trim to requested size
        final = deduped[:n_examples]

        return GenerationResult(
            task_type=self._task_type(),
            spec=spec.model_dump(),
            examples=final,
            total_generated=total_generated,
            total_after_critique=total_after_critique,
            total_after_dedup=total_after_dedup,
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
