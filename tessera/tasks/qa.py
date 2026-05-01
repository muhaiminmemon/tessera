"""QA task: orchestrates pipeline modules for question answering and RAG evaluation."""
from __future__ import annotations

import logging
from typing import Any

from tessera.core.base import TaskTemplate
from tessera.core.exceptions import ConfigurationError
from tessera.core.models import (
    Example,
    GenerationResult,
    Persona,
    QAExample,
    QAGenerationResult,
    QASpec,
    TaskSpec,
    TaskType,
    Taxonomy,
)
from tessera.pipeline.critique import CritiqueEngine
from tessera.pipeline.dedup import DedupEngine
from tessera.pipeline.generation import GenerationEngine
from tessera.pipeline.taxonomy import TaxonomyExpander

log = logging.getLogger(__name__)


class QATask(TaskTemplate):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        critique_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.85,
        critique_threshold: float = 7.5,
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
        return TaskType.QA

    def build_taxonomy(self, spec: TaskSpec) -> Taxonomy:
        if not isinstance(spec, QASpec):
            raise ConfigurationError(
                f"QATask requires QASpec, got {type(spec).__name__}"
            )
        return self._expander.expand(spec, TaskType.QA, model=self.model)

    def generate_example(
        self, node: Any, persona: Persona, spec: TaskSpec
    ) -> Example:
        if not isinstance(spec, QASpec):
            raise ConfigurationError(
                f"QATask requires QASpec, got {type(spec).__name__}"
            )
        results = self._generator.generate_batch(
            nodes=[node],
            personas=[persona],
            spec=spec,
            task_type=TaskType.QA,
            model=self.model,
            n=1,
        )
        if not results:
            raise RuntimeError("GenerationEngine returned no examples")
        return results[0]

    def critique_example(self, example: Example, spec: TaskSpec) -> Example:
        if not isinstance(spec, QASpec):
            raise ConfigurationError(
                f"QATask requires QASpec, got {type(spec).__name__}"
            )
        scores = self._critiquer.score(
            example=example,
            spec=spec,
            task_type=TaskType.QA,
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
            return [
                {
                    "context": ex.context,
                    "question": ex.question,
                    "answer": ex.answer,
                    "question_type": ex.question_type,
                    "difficulty": ex.difficulty,
                    "label": ex.label,
                }
                for ex in examples
            ]
        if fmt == "squad":
            rows = []
            for ex in examples:
                context = ex.context or ""
                answer = ex.answer or ""
                answer_start = context.find(answer)
                rows.append(
                    {
                        "id": ex.id,
                        "context": context,
                        "question": ex.question,
                        "answers": {
                            "text": [answer],
                            "answer_start": [max(answer_start, 0)],
                        },
                        "question_type": ex.question_type,
                        "difficulty": ex.difficulty,
                    }
                )
            return rows
        if fmt == "alpaca":
            return [
                {
                    "instruction": (
                        "Answer the question based only on the provided context. "
                        "If the answer cannot be determined from the context, respond with: "
                        "'This cannot be determined from the provided context.'"
                    ),
                    "input": f"Context: {ex.context}\n\nQuestion: {ex.question}",
                    "output": ex.answer,
                }
                for ex in examples
            ]
        raise ValueError(f"Unknown format '{fmt}'. Choose from: jsonl, squad, alpaca")

    def to_qa_result(self, result: GenerationResult) -> QAGenerationResult:
        """Convert a GenerationResult (internal Examples) to a typed QAGenerationResult."""
        qa_examples = [
            QAExample(
                id=ex.id,
                context=ex.context or "",
                question=ex.question or "",
                answer=ex.answer or "",
                question_type=ex.question_type or "",
                difficulty=ex.difficulty or "medium",
                label=ex.label or ex.question_type or "",
            )
            for ex in result.examples
        ]
        return QAGenerationResult(
            spec=result.spec,
            examples=qa_examples,
            total_generated=result.total_generated,
            total_after_critique=result.total_after_critique,
            total_after_dedup=result.total_after_dedup,
            cost_usd=result.cost_usd,
            metadata=result.metadata,
        )

    def validate_downstream(
        self, train: list[Example], test: list[Example]
    ) -> Any:
        raise NotImplementedError(
            "Downstream validation runs fine-tuning. "
            "Use tessera.validation.finetune.UnslothFinetuner and "
            "tessera.validation.evaluate.Evaluator instead."
        )
