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
        critique_threshold: float | None = None,
        max_attempts_multiplier: float = 2.5,
        target_per_label: int | None = None,
    ) -> GenerationResult:
        """End-to-end pipeline: taxonomy → generate → critique → dedup → trim."""
        taxonomy = self.build_taxonomy(spec)

        labels_in_taxonomy = sorted({n.target_label for n in taxonomy.nodes})
        num_labels = len(labels_in_taxonomy)
        effective_n = (
            target_per_label * num_labels
            if target_per_label is not None and num_labels > 0
            else n_examples
        )

        target_raw = int(effective_n * max_attempts_multiplier)
        nodes_sample = self._sample_nodes_balanced(taxonomy, target_raw)

        # Only override the task's critique_threshold when the caller explicitly sets one.
        if critique_threshold is not None and hasattr(self, "critique_threshold"):
            self.critique_threshold = critique_threshold

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

        if scored and len(passed) == len(scored):
            print(
                "[tessera] warning: 100% critique pass rate — "
                "consider raising threshold to 7.0"
            )

        # Fill-up pass: targeted generation for underrepresented labels
        if target_per_label is not None and num_labels > 0:
            passed = self._fill_up_labels(
                passed, taxonomy, labels_in_taxonomy, target_per_label,
                max_attempts_multiplier, personas, spec, _gen_worker, _critique_worker,
            )

        total_after_critique = len(passed)

        # Dedup
        deduped = self.deduplicate(passed)
        total_after_dedup = len(deduped)

        # Trim to requested size with strict label balance
        final = self._trim_to_balance(deduped, effective_n)

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

    def _fill_up_labels(
        self,
        passed: list[Example],
        taxonomy: Any,
        labels_in_taxonomy: list[str],
        target_per_label: int,
        max_attempts_multiplier: float,
        personas: list[Persona],
        spec: TaskSpec,
        gen_worker: Any,
        critique_worker: Any,
    ) -> list[Example]:
        """One extra generate+critique pass targeting labels below their quota."""
        from collections import Counter

        label_counts: Counter = Counter(ex.label for ex in passed if ex.label is not None)
        extra_nodes: list[Any] = []
        for lbl in labels_in_taxonomy:
            deficit = target_per_label - label_counts.get(lbl, 0)
            if deficit > 0:
                lbl_nodes = taxonomy.nodes_for_label(lbl)
                if lbl_nodes:
                    extra_nodes.extend(
                        random.choices(lbl_nodes, k=int(deficit * max_attempts_multiplier))
                    )

        if not extra_nodes:
            return passed

        max_workers = int(os.environ.get("TESSERA_MAX_CONCURRENT", "10"))
        print(f"[tessera] fill-up: {len(extra_nodes)} extra nodes for underrepresented labels")

        extra_raw: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for future in as_completed([executor.submit(gen_worker, nd) for nd in extra_nodes]):
                r = future.result()
                if r is not None:
                    extra_raw.append(r)

        extra_passed: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for future in as_completed([executor.submit(critique_worker, ex) for ex in extra_raw]):
                r = future.result()
                if r is not None and r.passed_critique:
                    extra_passed.append(r)

        return passed + extra_passed

    def _task_type(self) -> TaskType:
        raise NotImplementedError("subclass must override _task_type()")

    @staticmethod
    def _sample_nodes_balanced(taxonomy: Taxonomy, n: int) -> list[Any]:
        """Quota-based balanced node sampling.

        Each label gets exactly n // num_labels slots; remainder labels get one
        extra.  Within each quota, nodes are sampled with replacement so every
        label is always represented even when it has only one taxonomy node.
        """
        labels = sorted({node.target_label for node in taxonomy.nodes})
        if not labels:
            return []

        num_labels = len(labels)
        base_quota = n // num_labels
        remainder = n % num_labels
        buckets: dict[str, list[Any]] = {lbl: taxonomy.nodes_for_label(lbl) for lbl in labels}

        result: list[Any] = []
        for i, label in enumerate(labels):
            quota = base_quota + (1 if i < remainder else 0)
            bucket = buckets.get(label, [])
            if bucket:
                result.extend(random.choices(bucket, k=quota))

        random.shuffle(result)
        return result

    @staticmethod
    def _trim_to_balance(examples: list[Example], n: int) -> list[Example]:
        """Trim to n examples with exactly n // num_labels per label.

        Only applied for classification examples (those with a non-None label).
        Extraction / instruction examples fall back to a plain head-trim.
        """
        if not examples:
            return []

        # Non-classification tasks: simple trim
        if any(ex.label is None for ex in examples):
            return examples[:n]

        label_set = sorted({ex.label for ex in examples})  # type: ignore[arg-type]
        num_labels = len(label_set)
        if num_labels == 0:
            return examples[:n]

        buckets: dict[str, list[Example]] = {lbl: [] for lbl in label_set}
        for ex in examples:
            buckets[ex.label].append(ex)  # type: ignore[index]

        base_q = n // num_labels
        remainder = n % num_labels
        result: list[Example] = []
        for i, label in enumerate(label_set):
            quota = base_q + (1 if i < remainder else 0)
            result.extend(buckets[label][:quota])

        random.shuffle(result)
        return result
