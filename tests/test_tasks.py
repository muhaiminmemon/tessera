"""Tests for task classes — verifies contract, format methods, and type guards."""
from __future__ import annotations

import pytest

from tessera.core.exceptions import ConfigurationError
from tessera.core.models import Example, TaskType
from tessera.tasks.classification import ClassificationTask
from tessera.tasks.extraction import ExtractionTask
from tessera.tasks.instruction import InstructionTask
from tessera.tasks.qa import QATask

# ---------------------------------------------------------------------------
# ClassificationTask
# ---------------------------------------------------------------------------


class TestClassificationTask:
    def test_task_type(self) -> None:
        assert ClassificationTask()._task_type() == TaskType.CLASSIFICATION

    def test_build_taxonomy_rejects_wrong_spec(self, extraction_spec) -> None:
        task = ClassificationTask()
        with pytest.raises(ConfigurationError):
            task.build_taxonomy(extraction_spec)

    def test_format_jsonl(self, classification_spec) -> None:
        task = ClassificationTask()
        examples = [
            Example(task_type=TaskType.CLASSIFICATION, text="Hello", label="account_locked"),
            Example(task_type=TaskType.CLASSIFICATION, text="Help", label="card_lost"),
        ]
        rows = task.format_for_finetuning(examples, fmt="jsonl")
        assert rows[0] == {"text": "Hello", "label": "account_locked"}

    def test_format_alpaca(self, classification_spec) -> None:
        task = ClassificationTask()
        examples = [
            Example(task_type=TaskType.CLASSIFICATION, text="Hello", label="account_locked"),
        ]
        rows = task.format_for_finetuning(examples, fmt="alpaca")
        assert rows[0]["instruction"] == "Classify the following text."
        assert rows[0]["input"] == "Hello"
        assert rows[0]["output"] == "account_locked"

    def test_format_sharegpt(self) -> None:
        task = ClassificationTask()
        examples = [
            Example(task_type=TaskType.CLASSIFICATION, text="Hello", label="account_locked"),
        ]
        rows = task.format_for_finetuning(examples, fmt="sharegpt")
        assert rows[0]["conversations"][0]["from"] == "human"
        assert rows[0]["conversations"][1]["value"] == "account_locked"

    def test_format_unknown_raises(self) -> None:
        task = ClassificationTask()
        with pytest.raises(ValueError, match="Unknown format"):
            task.format_for_finetuning([], fmt="csv")

    def test_deduplicate_empty(self) -> None:
        task = ClassificationTask()
        assert task.deduplicate([]) == []


# ---------------------------------------------------------------------------
# ExtractionTask
# ---------------------------------------------------------------------------


class TestExtractionTask:
    def test_task_type(self) -> None:
        assert ExtractionTask()._task_type() == TaskType.EXTRACTION

    def test_build_taxonomy_rejects_wrong_spec(self, classification_spec) -> None:
        task = ExtractionTask()
        with pytest.raises(ConfigurationError):
            task.build_taxonomy(classification_spec)

    def test_format_jsonl(self) -> None:
        task = ExtractionTask()
        examples = [
            Example(
                task_type=TaskType.EXTRACTION,
                source_text="Contract between A and B",
                extracted_fields={"party_a": "A", "party_b": "B"},
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="jsonl")
        assert rows[0]["source_text"] == "Contract between A and B"
        assert rows[0]["extracted_fields"]["party_a"] == "A"

    def test_format_alpaca(self) -> None:
        task = ExtractionTask()
        examples = [
            Example(
                task_type=TaskType.EXTRACTION,
                source_text="Contract text",
                extracted_fields={"party_a": "A"},
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="alpaca")
        assert rows[0]["instruction"].startswith("Extract structured")
        assert "Contract text" in rows[0]["input"]

    def test_format_unknown_raises(self) -> None:
        task = ExtractionTask()
        with pytest.raises(ValueError, match="Unknown format"):
            task.format_for_finetuning([], fmt="csv")


# ---------------------------------------------------------------------------
# InstructionTask
# ---------------------------------------------------------------------------


class TestInstructionTask:
    def test_task_type(self) -> None:
        assert InstructionTask()._task_type() == TaskType.INSTRUCTION

    def test_build_taxonomy_rejects_wrong_spec(self, classification_spec) -> None:
        task = InstructionTask()
        with pytest.raises(ConfigurationError):
            task.build_taxonomy(classification_spec)

    def test_format_alpaca(self) -> None:
        task = InstructionTask()
        examples = [
            Example(
                task_type=TaskType.INSTRUCTION,
                instruction="Write a sort function",
                response="def sort(x): return sorted(x)",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="alpaca")
        assert rows[0]["instruction"] == "Write a sort function"
        assert rows[0]["output"] == "def sort(x): return sorted(x)"
        assert rows[0]["input"] == ""

    def test_format_jsonl(self) -> None:
        task = InstructionTask()
        examples = [
            Example(
                task_type=TaskType.INSTRUCTION,
                instruction="Explain recursion",
                response="It's a function that calls itself.",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="jsonl")
        assert rows[0]["instruction"] == "Explain recursion"

    def test_format_unknown_raises(self) -> None:
        task = InstructionTask()
        with pytest.raises(ValueError, match="Unknown format"):
            task.format_for_finetuning([], fmt="csv")


# ---------------------------------------------------------------------------
# QATask
# ---------------------------------------------------------------------------


class TestQATask:
    def test_task_type(self) -> None:
        assert QATask()._task_type() == TaskType.QA

    def test_build_taxonomy_rejects_wrong_spec(self, classification_spec) -> None:
        task = QATask()
        with pytest.raises(ConfigurationError):
            task.build_taxonomy(classification_spec)

    def test_format_jsonl(self) -> None:
        task = QATask()
        examples = [
            Example(
                task_type=TaskType.QA,
                context="Photosynthesis is the process...",
                question="What is photosynthesis?",
                answer="The process plants use to make food.",
                question_type="factoid",
                difficulty="easy",
                label="factoid",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="jsonl")
        assert rows[0]["question"] == "What is photosynthesis?"
        assert rows[0]["question_type"] == "factoid"

    def test_format_squad(self) -> None:
        task = QATask()
        context = "Photosynthesis converts sunlight into sugar."
        answer = "sunlight"
        examples = [
            Example(
                task_type=TaskType.QA,
                context=context,
                question="What does photosynthesis convert?",
                answer=answer,
                question_type="factoid",
                difficulty="easy",
                label="factoid",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="squad")
        assert rows[0]["answers"]["text"] == [answer]
        assert isinstance(rows[0]["answers"]["answer_start"][0], int)

    def test_format_alpaca(self) -> None:
        task = QATask()
        examples = [
            Example(
                task_type=TaskType.QA,
                context="Some context.",
                question="What?",
                answer="Something.",
                question_type="factoid",
                difficulty="easy",
                label="factoid",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="alpaca")
        assert "Context: Some context." in rows[0]["input"]
        assert rows[0]["output"] == "Something."

    def test_format_unknown_raises(self) -> None:
        task = QATask()
        with pytest.raises(ValueError, match="Unknown format"):
            task.format_for_finetuning([], fmt="csv")

    def test_to_qa_result(self, qa_spec) -> None:
        from tessera.core.models import GenerationResult

        task = QATask()
        gen_result = GenerationResult(
            task_type=TaskType.QA,
            spec=qa_spec.model_dump(),
            examples=[
                Example(
                    task_type=TaskType.QA,
                    context="ctx",
                    question="q?",
                    answer="a",
                    question_type="factoid",
                    difficulty="easy",
                    label="factoid",
                )
            ],
            total_generated=1,
            total_after_critique=1,
            total_after_dedup=1,
        )
        qa_result = task.to_qa_result(gen_result)
        assert qa_result.examples[0].question == "q?"
        assert qa_result.task_type == TaskType.QA
