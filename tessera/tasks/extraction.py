"""Extraction task: orchestrates pipeline modules for structured extraction."""
from __future__ import annotations

import json
from typing import Any

from tessera.core.base import TaskTemplate
from tessera.core.models import (
    Example,
    ExtractionSpec,
    Persona,
    TaskSpec,
    TaskType,
    Taxonomy,
)
from tessera.pipeline.critique import CritiqueEngine
from tessera.pipeline.dedup import DedupEngine
from tessera.pipeline.generation import GenerationEngine
from tessera.pipeline.taxonomy import TaxonomyExpander


class ExtractionTask(TaskTemplate):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        critique_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.90,
    ) -> None:
        self.model = model
        self.critique_model = critique_model
        self.dedup_threshold = dedup_threshold
        self._expander = TaxonomyExpander()
        self._generator = GenerationEngine()
        self._critiquer = CritiqueEngine()
        self._deduper = DedupEngine()

    def _task_type(self) -> TaskType:
        return TaskType.EXTRACTION

    def build_taxonomy(self, spec: TaskSpec) -> Taxonomy:
        assert isinstance(spec, ExtractionSpec)
        return self._expander.expand(spec, TaskType.EXTRACTION, model=self.model)

    def generate_example(
        self, node: Any, persona: Persona, spec: TaskSpec
    ) -> Example:
        assert isinstance(spec, ExtractionSpec)
        results = self._generator.generate_batch(
            nodes=[node],
            personas=[persona],
            spec=spec,
            task_type=TaskType.EXTRACTION,
            model=self.model,
            n=1,
        )
        if not results:
            raise RuntimeError("GenerationEngine returned no examples")
        return results[0]

    def critique_example(self, example: Example, spec: TaskSpec) -> Example:
        assert isinstance(spec, ExtractionSpec)
        scores = self._critiquer.score(
            example=example,
            spec=spec,
            task_type=TaskType.EXTRACTION,
            model=self.critique_model,
        )
        # For extraction: check field completeness alongside LLM scores
        all_fields_present = all(
            k in (example.extracted_fields or {})
            for k in spec.schema_definition
        )
        if not all_fields_present:
            scores = scores.model_copy(
                update={"label_correctness": min(scores.label_correctness, 4.0)}
            )
        example.critique_scores = scores
        example.passed_critique = scores.passes(6.0)
        return example

    def deduplicate(self, examples: list[Example]) -> list[Example]:
        return self._deduper.deduplicate(examples, threshold=self.dedup_threshold)

    def format_for_finetuning(
        self, examples: list[Example], fmt: str = "jsonl"
    ) -> list[dict[str, Any]]:
        if fmt == "jsonl":
            return [
                {
                    "source_text": ex.source_text,
                    "extracted_fields": ex.extracted_fields,
                }
                for ex in examples
            ]
        elif fmt in ("alpaca", "sharegpt"):
            rows = []
            for ex in examples:
                system = (
                    "Extract structured information from the provided text "
                    "and return valid JSON."
                )
                instruction = (
                    f"{ex.source_text}\n\n"
                    f"Extract all fields and return JSON."
                )
                output = json.dumps(ex.extracted_fields, ensure_ascii=False)
                if fmt == "alpaca":
                    rows.append(
                        {"instruction": system, "input": instruction, "output": output}
                    )
                else:
                    rows.append(
                        {
                            "conversations": [
                                {"from": "system", "value": system},
                                {"from": "human", "value": instruction},
                                {"from": "gpt", "value": output},
                            ]
                        }
                    )
            return rows
        else:
            raise ValueError(f"Unknown format '{fmt}'. Choose from: jsonl, alpaca, sharegpt")

    def validate_downstream(
        self, train: list[Example], test: list[Example]
    ) -> Any:
        raise NotImplementedError(
            "Downstream validation runs fine-tuning. "
            "Use tessera.validation.finetune.UnslothFinetuner and "
            "tessera.validation.evaluate.Evaluator instead."
        )
