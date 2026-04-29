"""Unit tests for Pydantic models — no LLM calls required."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tessera.core.models import (
    ClassificationSpec,
    CritiqueScores,
    Example,
    Persona,
    Taxonomy,
    TaxonomyNode,
    TaskType,
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

    def test_personas_list_has_50(self) -> None:
        assert len(PERSONAS) == 50

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
