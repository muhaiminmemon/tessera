"""Expands a task spec into a Taxonomy of generation-ready nodes."""
from __future__ import annotations

import json
import logging
import warnings
from typing import Callable

from tessera.core import config as _cfg
from tessera.core import prompts
from tessera.core.exceptions import ConfigurationError, TaxonomyError
from tessera.core.llm_client import get_client
from tessera.core.models import (
    ClassificationSpec,
    ExtractionSpec,
    InstructionSpec,
    QASpec,
    TaskSpec,
    TaskType,
    Taxonomy,
    TaxonomyNode,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dispatch tables — adding a new task type only requires entries here, not
# changes inside TaxonomyExpander.expand (Open/Closed Principle).
# ---------------------------------------------------------------------------

_PromptPair = tuple[Callable[..., str], Callable[..., str]]

_TAXONOMY_PROMPTS: dict[TaskType, _PromptPair] = {
    TaskType.CLASSIFICATION: (
        prompts.classification_taxonomy_system,
        prompts.classification_taxonomy_user,
    ),
    TaskType.EXTRACTION: (
        prompts.extraction_taxonomy_system,
        prompts.extraction_taxonomy_user,
    ),
    TaskType.INSTRUCTION: (
        prompts.instruction_taxonomy_system,
        prompts.instruction_taxonomy_user,
    ),
    TaskType.QA: (
        prompts.qa_taxonomy_system,
        prompts.qa_taxonomy_user,
    ),
}

# Maps each task_type to a function that extracts the list of labels to verify
# coverage for.  Task types with no label coverage check are absent.
_COVERAGE_EXTRACTORS: dict[TaskType, Callable[[TaskSpec], list[str]]] = {
    TaskType.CLASSIFICATION: lambda s: s.labels,  # type: ignore[union-attr]
    TaskType.INSTRUCTION: lambda s: s.instruction_types,  # type: ignore[union-attr]
    TaskType.QA: lambda s: s.question_types,  # type: ignore[union-attr]
}

# Expected spec types per task — used for type-safe validation.
_EXPECTED_SPEC: dict[TaskType, type] = {
    TaskType.CLASSIFICATION: ClassificationSpec,
    TaskType.EXTRACTION: ExtractionSpec,
    TaskType.INSTRUCTION: InstructionSpec,
    TaskType.QA: QASpec,
}


class TaxonomyExpander:
    def expand(
        self,
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
    ) -> Taxonomy:
        expected = _EXPECTED_SPEC.get(task_type)
        if expected is None:
            raise ConfigurationError(f"Unknown task_type: {task_type}")
        if not isinstance(spec, expected):
            raise ConfigurationError(
                f"Expected {expected.__name__} for {task_type}, "
                f"got {type(spec).__name__}"
            )

        if task_type not in _TAXONOMY_PROMPTS:
            raise ConfigurationError(f"No prompt builder registered for {task_type}")

        sys_fn, usr_fn = _TAXONOMY_PROMPTS[task_type]
        sys_msg = sys_fn(spec)
        usr_msg = usr_fn(spec)

        client = get_client()
        raw = client.complete(
            model=model,
            system=sys_msg,
            user=usr_msg,
            temperature=_cfg.TAXONOMY_TEMPERATURE,
            max_tokens=4096,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TaxonomyError(f"LLM returned invalid JSON for taxonomy: {exc}") from exc

        nodes_data = data.get("nodes", [])
        nodes: list[TaxonomyNode] = []
        for nd in nodes_data:
            try:
                nodes.append(
                    TaxonomyNode(
                        label=nd.get("label", ""),
                        category=nd.get("category", ""),
                        subcategory=nd.get("subcategory", ""),
                        scenario=nd.get("scenario", ""),
                        depth=int(nd.get("depth", 1)),
                        target_label=nd.get("target_label", ""),
                    )
                )
            except Exception as exc:
                warnings.warn(f"[TaxonomyExpander] skipped malformed node: {exc}")

        if not nodes:
            raise TaxonomyError(
                f"TaxonomyExpander returned 0 nodes for task_type={task_type}. "
                "Try increasing max_tokens or rephrasing the domain."
            )

        taxonomy = Taxonomy(task_type=task_type, nodes=nodes)
        log.info("taxonomy expanded: %d nodes for %s", len(nodes), task_type)

        coverage_fn = _COVERAGE_EXTRACTORS.get(task_type)
        if coverage_fn is not None:
            for label in coverage_fn(spec):
                if not taxonomy.nodes_for_label(label):
                    warnings.warn(
                        f"[TaxonomyExpander] label '{label}' has no taxonomy nodes. "
                        "Consider re-running or increasing max_tokens."
                    )

        return taxonomy
