"""Abstract base class that all task implementations inherit from."""
from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from tessera.core import config as _cfg
from tessera.core.exceptions import ConfigurationError
from tessera.core.llm_client import get_client
from tessera.core.models import (
    Example,
    GenerationResult,
    Persona,
    TaskSpec,
    TaskType,
    Taxonomy,
)

log = logging.getLogger(__name__)


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

    @abstractmethod
    def _task_type(self) -> TaskType:
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
        max_retries: int = 3,
    ) -> GenerationResult:
        """End-to-end pipeline: taxonomy → generate → critique → dedup → trim.

        Guarantees the requested ``n_examples`` through a retry loop: after each
        generate→critique→dedup cycle, if the surviving pool is smaller than the
        target the pipeline generates exactly the deficit (× multiplier) more
        examples and merges them into the pool.  Retries up to ``max_retries``
        times before logging a warning and returning whatever was collected.
        """
        if n_examples <= 0:
            raise ConfigurationError(f"n_examples must be > 0, got {n_examples}")
        if not personas:
            raise ConfigurationError("personas list must not be empty")
        if max_retries < 0:
            raise ConfigurationError(f"max_retries must be >= 0, got {max_retries}")

        taxonomy = self.build_taxonomy(spec)
        labels_in_taxonomy = sorted({n.target_label for n in taxonomy.nodes})
        num_labels = len(labels_in_taxonomy)

        effective_n = (
            target_per_label * num_labels
            if target_per_label is not None and num_labels > 0
            else n_examples
        )

        if critique_threshold is not None and hasattr(self, "critique_threshold"):
            self.critique_threshold = critique_threshold

        max_workers = _cfg.max_concurrent()
        cost_before = get_client().usage.cost_usd

        # --- Initial generation pass (2.5× oversampling) ---
        target_raw = int(effective_n * max_attempts_multiplier)
        nodes_sample = self._sample_nodes_balanced(taxonomy, target_raw)
        raw_examples = self._run_generation_phase(nodes_sample, personas, spec, max_workers)
        total_generated = len(raw_examples)

        passed = self._run_critique_phase(raw_examples, spec, max_workers)

        if target_per_label is not None and num_labels > 0:
            passed = self._fill_up_labels(
                passed, taxonomy, labels_in_taxonomy, target_per_label,
                max_attempts_multiplier, personas, spec, max_workers,
            )

        total_after_critique = len(passed)

        # --- Retry loop: keep topping up until we have enough ---
        deduped = self.deduplicate(passed)
        attempt = 0
        while len(deduped) < effective_n and attempt < max_retries:
            deficit = effective_n - len(deduped)
            attempt += 1
            log.info(
                "top-up retry %d/%d — need %d more examples after dedup",
                attempt, max_retries, deficit,
            )
            top_up_nodes = self._sample_nodes_balanced(
                taxonomy, int(deficit * max_attempts_multiplier)
            )
            top_up_raw = self._run_generation_phase(
                top_up_nodes, personas, spec, max_workers
            )
            top_up_passed = self._run_critique_phase(top_up_raw, spec, max_workers)
            total_generated += len(top_up_raw)
            total_after_critique += len(top_up_passed)
            # Re-dedup the combined pool so new examples don't duplicate existing ones
            deduped = self.deduplicate(deduped + top_up_passed)

        if len(deduped) < effective_n:
            log.warning(
                "could only collect %d/%d examples after %d retries — "
                "try lowering critique_threshold or dedup_threshold",
                len(deduped), effective_n, max_retries,
            )

        total_after_dedup = len(deduped)
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
    # Pipeline phase helpers
    # ------------------------------------------------------------------

    def _run_generation_phase(
        self,
        nodes_sample: list[Any],
        personas: list[Persona],
        spec: TaskSpec,
        max_workers: int,
    ) -> list[Example]:
        """Parallel generation phase; returns all successfully generated examples."""
        def _gen_worker(node: Any) -> Example | None:
            persona = random.choice(personas)
            try:
                return self.generate_example(node, persona, spec)
            except Exception as exc:
                log.warning("generation failed for node %s: %s", node.id, exc)
                return None

        raw: list[Example] = []
        total = len(nodes_sample)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_gen_worker, node) for node in nodes_sample]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    raw.append(result)
                if i % 10 == 0:
                    log.info("generating %d/%d...", i, total)

        return raw

    def _run_critique_phase(
        self,
        raw_examples: list[Example],
        spec: TaskSpec,
        max_workers: int,
    ) -> list[Example]:
        """Parallel critique phase; returns examples that passed the threshold."""
        def _critique_worker(ex: Example) -> Example | None:
            try:
                return self.critique_example(ex, spec)
            except Exception as exc:
                log.warning("critique failed for example %s: %s", ex.id, exc)
                return None

        scored: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_critique_worker, ex): ex for ex in raw_examples}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    scored.append(result)

        passed = [ex for ex in scored if ex.passed_critique]

        if scored and len(passed) == len(scored):
            log.warning(
                "100%% critique pass rate — consider raising critique_threshold to 7.0"
            )

        return passed

    # ------------------------------------------------------------------
    # Fill-up and balancing helpers
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
        max_workers: int,
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

        log.info("fill-up: %d extra nodes for underrepresented labels", len(extra_nodes))
        extra_raw = self._run_generation_phase(extra_nodes, personas, spec, max_workers)
        extra_passed = self._run_critique_phase(extra_raw, spec, max_workers)
        return passed + extra_passed

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
