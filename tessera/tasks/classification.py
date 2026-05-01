"""Classification task: orchestrates pipeline modules for text classification."""
from __future__ import annotations

import logging
import json
from typing import Any

from tessera.core.base import TaskTemplate
from tessera.core.exceptions import ConfigurationError
from tessera.core.models import (
    ClassificationSpec,
    Example,
    Persona,
    TaskSpec,
    TaskType,
    Taxonomy,
)
from tessera.pipeline.critique import CritiqueEngine
from tessera.pipeline.dedup import DedupEngine
from tessera.pipeline.generation import GenerationEngine
from tessera.pipeline.taxonomy import TaxonomyExpander

log = logging.getLogger(__name__)


class ClassificationTask(TaskTemplate):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        critique_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.85,
        critique_threshold: float = 7.0,
    ) -> None:
        self.model = model
        self.critique_model = critique_model
        self.dedup_threshold = dedup_threshold
        self.critique_threshold = critique_threshold
        self._expander = TaxonomyExpander()
        self._generator = GenerationEngine()
        self._critiquer = CritiqueEngine()
        self._deduper = DedupEngine()

    def _task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def build_taxonomy(self, spec: TaskSpec) -> Taxonomy:
        if not isinstance(spec, ClassificationSpec):
            raise ConfigurationError(
                f"ClassificationTask requires ClassificationSpec, got {type(spec).__name__}"
            )
        return self._expander.expand(spec, TaskType.CLASSIFICATION, model=self.model)

    def generate_example(
        self, node: Any, persona: Persona, spec: TaskSpec
    ) -> Example:
        if not isinstance(spec, ClassificationSpec):
            raise ConfigurationError(
                f"ClassificationTask requires ClassificationSpec, got {type(spec).__name__}"
            )
        results = self._generator.generate_batch(
            nodes=[node],
            personas=[persona],
            spec=spec,
            task_type=TaskType.CLASSIFICATION,
            model=self.model,
            n=1,
        )
        if not results:
            raise RuntimeError("GenerationEngine returned no examples")
        return results[0]

    def critique_example(self, example: Example, spec: TaskSpec) -> Example:
        if not isinstance(spec, ClassificationSpec):
            raise ConfigurationError(
                f"ClassificationTask requires ClassificationSpec, got {type(spec).__name__}"
            )
        scores = self._critiquer.score(
            example=example,
            spec=spec,
            task_type=TaskType.CLASSIFICATION,
            model=self.critique_model,
        )
        example.critique_scores = scores
        example.passed_critique = scores.passes(self.critique_threshold)
        return example

    def deduplicate(self, examples: list[Example]) -> list[Example]:
        return self._deduper.deduplicate(examples, threshold=self.dedup_threshold)

    def format_for_finetuning(
        self, examples: list[Example], fmt: str = "jsonl"
    ) -> list[dict[str, Any]]:
        if fmt == "jsonl":
            return [{"text": ex.text, "label": ex.label} for ex in examples]
        if fmt == "alpaca":
            return [
                {
                    "instruction": "Classify the following text.",
                    "input": ex.text,
                    "output": ex.label,
                }
                for ex in examples
            ]
        if fmt == "sharegpt":
            return [
                {
                    "conversations": [
                        {"from": "human", "value": f"Classify this text: {ex.text}"},
                        {"from": "gpt", "value": ex.label},
                    ]
                }
                for ex in examples
            ]
        raise ValueError(f"Unknown format '{fmt}'. Choose from: jsonl, alpaca, sharegpt")

    def validate_downstream(
        self, train: list[Example], test: list[Example]
    ) -> Any:
        raise NotImplementedError(
            "Downstream validation runs fine-tuning. "
            "Use tessera.validation.finetune.UnslothFinetuner and "
            "tessera.validation.evaluate.Evaluator instead."
        )
