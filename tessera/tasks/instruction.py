"""Instruction task: orchestrates pipeline modules for instruction-following."""
from __future__ import annotations

from typing import Any

from tessera.core.base import TaskTemplate
from tessera.core.models import (
    Example,
    InstructionSpec,
    Persona,
    TaskSpec,
    TaskType,
    Taxonomy,
)
from tessera.pipeline.critique import CritiqueEngine
from tessera.pipeline.dedup import DedupEngine
from tessera.pipeline.generation import GenerationEngine
from tessera.pipeline.taxonomy import TaxonomyExpander


class InstructionTask(TaskTemplate):
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
        return TaskType.INSTRUCTION

    def build_taxonomy(self, spec: TaskSpec) -> Taxonomy:
        assert isinstance(spec, InstructionSpec)
        return self._expander.expand(spec, TaskType.INSTRUCTION, model=self.model)

    def generate_example(
        self, node: Any, persona: Persona, spec: TaskSpec
    ) -> Example:
        assert isinstance(spec, InstructionSpec)
        results = self._generator.generate_batch(
            nodes=[node],
            personas=[persona],
            spec=spec,
            task_type=TaskType.INSTRUCTION,
            model=self.model,
            n=1,
        )
        if not results:
            raise RuntimeError("GenerationEngine returned no examples")
        return results[0]

    def critique_example(self, example: Example, spec: TaskSpec) -> Example:
        assert isinstance(spec, InstructionSpec)
        scores = self._critiquer.score(
            example=example,
            spec=spec,
            task_type=TaskType.INSTRUCTION,
            model=self.critique_model,
        )
        example.critique_scores = scores
        example.passed_critique = scores.passes(6.0)
        return example

    def deduplicate(self, examples: list[Example]) -> list[Example]:
        return self._deduper.deduplicate(examples, threshold=self.dedup_threshold)

    def format_for_finetuning(
        self, examples: list[Example], fmt: str = "jsonl"
    ) -> list[dict[str, Any]]:
        if fmt == "alpaca":
            return [
                {
                    "instruction": ex.instruction,
                    "input": "",
                    "output": ex.response,
                }
                for ex in examples
            ]
        elif fmt == "sharegpt":
            return [
                {
                    "conversations": [
                        {"from": "human", "value": ex.instruction},
                        {"from": "gpt", "value": ex.response},
                    ]
                }
                for ex in examples
            ]
        elif fmt == "jsonl":
            return [
                {"instruction": ex.instruction, "response": ex.response}
                for ex in examples
            ]
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
