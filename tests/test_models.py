"""Unit tests for Pydantic models — no LLM calls required."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tessera.core.models import (
    ClassificationSpec,
    CritiqueScores,
    Example,
    Persona,
    QAExample,
    QAGenerationResult,
    QASpec,
    TaskType,
    Taxonomy,
    TaxonomyNode,
)
from tessera.core.personas import PERSONAS


class TestClassificationSpec:
    def test_valid_spec(self) -> None:
        spec = ClassificationSpec(
            domain="banking",
            labels=["positive", "negative", "neutral"],
        )
        assert len(spec.labels) == 3
        assert spec.language == "English"

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            ClassificationSpec(domain="test", labels=["a", "a", "b"])

    def test_too_few_labels_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClassificationSpec(domain="test", labels=["only_one"])

    def test_label_descriptions_optional(self) -> None:
        spec = ClassificationSpec(domain="test", labels=["x", "y"])
        assert spec.label_descriptions == {}


class TestExample:
    def test_valid_classification_example(self) -> None:
        ex = Example(task_type=TaskType.CLASSIFICATION, text="hello world", label="positive")
        assert ex.text == "hello world"
        assert ex.label == "positive"
        assert ex.passed_critique is False

    def test_missing_text_rejected(self) -> None:
        with pytest.raises(ValidationError, match="text is required"):
            Example(task_type=TaskType.CLASSIFICATION, label="positive")

    def test_missing_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label is required"):
            Example(task_type=TaskType.CLASSIFICATION, text="some text")

    def test_valid_extraction_example(self) -> None:
        ex = Example(
            task_type=TaskType.EXTRACTION,
            source_text="Alice works at Acme Corp.",
            extracted_fields={"person": "Alice", "org": "Acme Corp"},
        )
        assert ex.extracted_fields["person"] == "Alice"

    def test_missing_source_text_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_text is required"):
            Example(
                task_type=TaskType.EXTRACTION,
                extracted_fields={"field": "value"},
            )

    def test_valid_instruction_example(self) -> None:
        ex = Example(
            task_type=TaskType.INSTRUCTION,
            instruction="Write a hello world function.",
            response="def hello():\n    print('Hello, world!')",
        )
        assert "Hello" in ex.response

    def test_missing_instruction_rejected(self) -> None:
        with pytest.raises(ValidationError, match="instruction is required"):
            Example(task_type=TaskType.INSTRUCTION, response="some response")


class TestCritiqueScores:
    def test_mean_calculation(self) -> None:
        scores = CritiqueScores(realism=6.0, label_correctness=8.0, specificity=7.0)
        assert abs(scores.mean - 7.0) < 1e-9

    def test_passes_threshold(self) -> None:
        scores = CritiqueScores(realism=7.0, label_correctness=7.0, specificity=7.0)
        assert scores.passes(6.0) is True
        assert scores.passes(7.5) is False

    def test_boundary_passes(self) -> None:
        scores = CritiqueScores(realism=6.0, label_correctness=6.0, specificity=6.0)
        assert scores.passes(6.0) is True

    def test_out_of_range_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CritiqueScores(realism=11.0, label_correctness=5.0, specificity=5.0)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CritiqueScores(realism=-1.0, label_correctness=5.0, specificity=5.0)


class TestPersona:
    def test_to_prompt_fragment_contains_expected_fields(self) -> None:
        persona = Persona(
            name="Test User",
            formality="casual",
            expertise="novice",
            age_range="20-30",
            cultural_context="North American",
            writing_quirks=["uses emoji", "lowercase"],
        )
        fragment = persona.to_prompt_fragment()
        assert "Test User" in fragment
        assert "casual" in fragment
        assert "novice" in fragment
        assert "20-30" in fragment
        assert "North American" in fragment
        assert "uses emoji" in fragment

    def test_empty_writing_quirks(self) -> None:
        persona = Persona(
            name="Plain",
            formality="formal",
            expertise="expert",
            age_range="35-45",
            cultural_context="UK",
        )
        fragment = persona.to_prompt_fragment()
        assert "none" in fragment

    def test_personas_list_has_60(self) -> None:
        assert len(PERSONAS) == 60

    def test_all_personas_have_required_fields(self) -> None:
        for p in PERSONAS:
            assert p.name
            assert p.formality in ("formal", "semi-formal", "casual", "very_casual")
            assert p.expertise in ("expert", "intermediate", "novice", "none")
            assert p.age_range
            assert p.cultural_context


class TestTaxonomy:
    def _make_taxonomy(self) -> Taxonomy:
        nodes = [
            TaxonomyNode(
                label="Positive sentiment",
                category="Sentiment",
                subcategory="Positive",
                scenario="User is happy",
                target_label="positive",
            ),
            TaxonomyNode(
                label="Positive sentiment 2",
                category="Sentiment",
                subcategory="Positive",
                scenario="User expresses joy",
                target_label="positive",
            ),
            TaxonomyNode(
                label="Negative sentiment",
                category="Sentiment",
                subcategory="Negative",
                scenario="User is upset",
                target_label="negative",
            ),
        ]
        return Taxonomy(task_type=TaskType.CLASSIFICATION, nodes=nodes)

    def test_nodes_for_label_filters_correctly(self) -> None:
        taxonomy = self._make_taxonomy()
        positive_nodes = taxonomy.nodes_for_label("positive")
        assert len(positive_nodes) == 2
        assert all(n.target_label == "positive" for n in positive_nodes)

    def test_nodes_for_label_missing_label(self) -> None:
        taxonomy = self._make_taxonomy()
        assert taxonomy.nodes_for_label("nonexistent") == []

    def test_len(self) -> None:
        taxonomy = self._make_taxonomy()
        assert len(taxonomy) == 3

    def test_empty_taxonomy(self) -> None:
        taxonomy = Taxonomy(task_type=TaskType.CLASSIFICATION)
        assert len(taxonomy) == 0
        assert taxonomy.nodes_for_label("x") == []


class TestQASpec:
    def test_valid_spec_default_question_types(self) -> None:
        spec = QASpec(domain="medical records")
        assert set(spec.question_types) == {"factoid", "multi-hop", "abstractive", "unanswerable"}
        assert spec.language == "English"

    def test_valid_spec_subset_question_types(self) -> None:
        spec = QASpec(domain="legal contracts", question_types=["factoid", "unanswerable"])
        assert spec.question_types == ["factoid", "unanswerable"]

    def test_invalid_question_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Invalid question_types"):
            QASpec(domain="test", question_types=["factoid", "open-ended"])

    def test_empty_question_types_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QASpec(domain="test", question_types=[])

    def test_single_question_type_allowed(self) -> None:
        spec = QASpec(domain="finance", question_types=["multi-hop"])
        assert spec.question_types == ["multi-hop"]


class TestQAExample:
    def test_valid_qa_example(self) -> None:
        ex = QAExample(
            context="The Eiffel Tower was built in 1889.",
            question="When was the Eiffel Tower built?",
            answer="1889",
            question_type="factoid",
            difficulty="easy",
            label="factoid",
        )
        assert ex.context == "The Eiffel Tower was built in 1889."
        assert ex.question_type == "factoid"
        assert ex.difficulty == "easy"
        assert ex.label == "factoid"

    def test_default_difficulty_is_medium(self) -> None:
        ex = QAExample(
            context="Some passage.",
            question="What is it?",
            answer="It is something.",
            question_type="abstractive",
            label="abstractive",
        )
        assert ex.difficulty == "medium"

    def test_id_is_auto_generated(self) -> None:
        ex = QAExample(
            context="ctx",
            question="q?",
            answer="a",
            question_type="factoid",
            label="factoid",
        )
        assert ex.id and len(ex.id) > 0

    def test_squad_format_conversion(self) -> None:
        """Verify the SQuAD-style dict structure produced by QATask.format_for_finetuning."""
        from tessera.tasks.qa import QATask

        task = QATask()
        context = "Paris is the capital of France."
        answer = "Paris"
        examples = [
            Example(
                task_type=TaskType.QA,
                context=context,
                question="What is the capital of France?",
                answer=answer,
                question_type="factoid",
                difficulty="easy",
                label="factoid",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="squad")
        assert len(rows) == 1
        row = rows[0]
        assert row["context"] == context
        assert row["answers"]["text"] == [answer]
        assert isinstance(row["answers"]["answer_start"], list)
        assert row["answers"]["answer_start"][0] == context.find(answer)

    def test_alpaca_format_conversion(self) -> None:
        from tessera.tasks.qa import QATask

        task = QATask()
        examples = [
            Example(
                task_type=TaskType.QA,
                context="Oxygen has atomic number 8.",
                question="What is the atomic number of oxygen?",
                answer="8",
                question_type="factoid",
                difficulty="easy",
                label="factoid",
            )
        ]
        rows = task.format_for_finetuning(examples, fmt="alpaca")
        assert len(rows) == 1
        assert "instruction" in rows[0]
        assert "Context:" in rows[0]["input"]
        assert rows[0]["output"] == "8"

    def test_unknown_format_raises(self) -> None:
        from tessera.tasks.qa import QATask

        task = QATask()
        with pytest.raises(ValueError, match="Unknown format"):
            task.format_for_finetuning([], fmt="xml")


class TestQAExampleValidation:
    def test_invalid_question_type_on_example(self) -> None:
        with pytest.raises(ValidationError, match="question_type must be one of"):
            Example(
                task_type=TaskType.QA,
                context="Some context.",
                question="What happened?",
                answer="Something.",
                question_type="open-book",
            )

    def test_invalid_difficulty_on_example(self) -> None:
        with pytest.raises(ValidationError, match="difficulty must be one of"):
            Example(
                task_type=TaskType.QA,
                context="Some context.",
                question="What happened?",
                answer="Something.",
                question_type="factoid",
                difficulty="trivial",
            )

    def test_missing_context_rejected(self) -> None:
        with pytest.raises(ValidationError, match="context is required"):
            Example(
                task_type=TaskType.QA,
                question="What happened?",
                answer="Something happened.",
            )

    def test_missing_question_rejected(self) -> None:
        with pytest.raises(ValidationError, match="question is required"):
            Example(
                task_type=TaskType.QA,
                context="Some context.",
                answer="Something happened.",
            )

    def test_missing_answer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="answer is required"):
            Example(
                task_type=TaskType.QA,
                context="Some context.",
                question="What happened?",
            )

    def test_valid_unanswerable_example(self) -> None:
        ex = Example(
            task_type=TaskType.QA,
            context="The meeting was held on Tuesday.",
            question="Who attended the meeting?",
            answer="This cannot be determined from the provided context.",
            question_type="unanswerable",
            difficulty="hard",
            label="unanswerable",
        )
        assert ex.question_type == "unanswerable"
        assert ex.difficulty == "hard"

    def test_qa_generation_result_schema(self) -> None:
        qa_ex = QAExample(
            context="ctx",
            question="q?",
            answer="a",
            question_type="multi-hop",
            label="multi-hop",
        )
        result = QAGenerationResult(
            spec={"domain": "test"},
            examples=[qa_ex],
            total_generated=10,
            total_after_critique=8,
            total_after_dedup=7,
        )
        assert result.task_type == TaskType.QA
        assert len(result.examples) == 1
        assert result.total_generated == 10
        assert result.cost_usd == 0.0
