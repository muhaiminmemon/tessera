"""Tests for pipeline modules — all LLM calls are mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tessera.core.exceptions import ConfigurationError, GenerationError, TaxonomyError
from tessera.core.models import (
    ClassificationSpec,
    Example,
    ExtractionSpec,
    TaskType,
    TaxonomyNode,
)
from tessera.pipeline.critique import CritiqueEngine, _parse_scores
from tessera.pipeline.dedup import DedupEngine
from tessera.pipeline.generation import GenerationEngine, _parse_json
from tessera.pipeline.taxonomy import TaxonomyExpander

# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_plain_json(self) -> None:
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_strips_json_fence(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        assert _parse_json(raw) == {"key": "value"}

    def test_strips_plain_fence(self) -> None:
        raw = '```\n{"key": "value"}\n```'
        assert _parse_json(raw) == {"key": "value"}

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(GenerationError, match="LLM returned invalid JSON"):
            _parse_json("not json")


# ---------------------------------------------------------------------------
# _parse_scores
# ---------------------------------------------------------------------------


class TestParseScores:
    def test_standard_fields(self) -> None:
        data = {"realism": 8.0, "label_correctness": 7.5, "specificity": 9.0, "reasoning": "ok"}
        scores = _parse_scores(TaskType.CLASSIFICATION, data)
        assert scores.realism == 8.0
        assert scores.label_correctness == 7.5
        assert scores.specificity == 9.0

    def test_qa_field_mapping(self) -> None:
        data = {
            "groundedness": 9.0,
            "question_clarity": 8.5,
            "answer_completeness": 7.0,
            "reasoning": "good",
        }
        scores = _parse_scores(TaskType.QA, data)
        assert scores.realism == 9.0
        assert scores.label_correctness == 8.5
        assert scores.specificity == 7.0

    def test_missing_keys_default_to_zero(self) -> None:
        scores = _parse_scores(TaskType.CLASSIFICATION, {})
        assert scores.realism == 0.0
        assert scores.mean == 0.0


# ---------------------------------------------------------------------------
# DedupEngine
# ---------------------------------------------------------------------------


class TestDedupEngine:
    def test_single_example_returned_as_is(self, classification_spec) -> None:
        engine = DedupEngine()
        ex = Example(
            task_type=TaskType.CLASSIFICATION,
            text="Hello world",
            label="account_locked",
        )
        result = engine.deduplicate([ex])
        assert result == [ex]

    def test_empty_list(self) -> None:
        engine = DedupEngine()
        assert engine.deduplicate([]) == []

    def test_example_text_classification(self) -> None:
        engine = DedupEngine()
        ex = Example(task_type=TaskType.CLASSIFICATION, text="some text", label="lbl")
        assert engine._example_text(ex) == "some text"

    def test_example_text_qa_uses_question_only(self) -> None:
        engine = DedupEngine()
        ex = Example(
            task_type=TaskType.QA,
            context="Long passage about science...",
            question="What is gravity?",
            answer="A force.",
        )
        assert engine._example_text(ex) == "What is gravity?"

    def test_example_text_extraction(self) -> None:
        engine = DedupEngine()
        ex = Example(
            task_type=TaskType.EXTRACTION,
            source_text="Contract between A and B",
            extracted_fields={"party_a": "A"},
        )
        assert engine._example_text(ex) == "Contract between A and B"

    def test_example_text_instruction(self) -> None:
        engine = DedupEngine()
        ex = Example(
            task_type=TaskType.INSTRUCTION,
            instruction="Write a sort function",
            response="def sort(x): return sorted(x)",
        )
        text = engine._example_text(ex)
        assert "Write a sort function" in text


# ---------------------------------------------------------------------------
# TaxonomyExpander
# ---------------------------------------------------------------------------


class TestTaxonomyExpander:
    def test_unknown_task_type_raises(self) -> None:
        expander = TaxonomyExpander()
        spec = ClassificationSpec(domain="x", labels=["a", "b"])
        with pytest.raises(ConfigurationError):
            expander.expand(spec, "invalid_task_type")  # type: ignore[arg-type]

    def test_wrong_spec_type_raises(self) -> None:
        expander = TaxonomyExpander()
        wrong_spec = ExtractionSpec(domain="x", schema_definition={"f": "str"})
        with pytest.raises(ConfigurationError):
            expander.expand(wrong_spec, TaskType.CLASSIFICATION)

    @patch("tessera.pipeline.taxonomy.get_client")
    def test_empty_nodes_raises_taxonomy_error(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"nodes": []})
        mock_get_client.return_value = mock_client

        expander = TaxonomyExpander()
        spec = ClassificationSpec(domain="banking", labels=["a", "b"])
        with pytest.raises(TaxonomyError):
            expander.expand(spec, TaskType.CLASSIFICATION)

    @patch("tessera.pipeline.taxonomy.get_client")
    def test_successful_expansion(self, mock_get_client) -> None:
        nodes_data = [
            {
                "label": "account_locked",
                "category": "banking",
                "subcategory": "access",
                "scenario": "Customer locked out",
                "depth": 1,
                "target_label": "account_locked",
            }
        ]
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"nodes": nodes_data})
        mock_get_client.return_value = mock_client

        expander = TaxonomyExpander()
        spec = ClassificationSpec(domain="banking", labels=["account_locked", "balance_inquiry"])
        taxonomy = expander.expand(spec, TaskType.CLASSIFICATION)
        assert len(taxonomy.nodes) == 1
        assert taxonomy.nodes[0].target_label == "account_locked"


# ---------------------------------------------------------------------------
# GenerationEngine
# ---------------------------------------------------------------------------


class TestGenerationEngine:
    @patch("tessera.pipeline.generation.get_client")
    def test_generate_classification_example(
        self, mock_get_client, sample_persona, classification_spec
    ) -> None:
        from tessera.core.llm_client import LLMClient

        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = json.dumps(
            {"text": "I cannot access my account", "label": "account_locked"}
        )
        mock_get_client.return_value = mock_client

        engine = GenerationEngine()
        node = TaxonomyNode(
            label="account_locked",
            category="banking",
            subcategory="access",
            scenario="Customer locked out",
            target_label="account_locked",
        )
        examples = engine.generate_batch(
            nodes=[node],
            personas=[sample_persona],
            spec=classification_spec,
            task_type=TaskType.CLASSIFICATION,
            n=1,
        )
        assert len(examples) == 1
        assert examples[0].text == "I cannot access my account"
        assert examples[0].label == "account_locked"

    @patch("tessera.pipeline.generation.get_client")
    def test_unknown_task_type_logs_warning(
        self, mock_get_client, sample_persona, classification_spec
    ) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({"text": "x", "label": "a"})
        mock_get_client.return_value = mock_client

        engine = GenerationEngine()
        node = TaxonomyNode(
            label="a",
            category="x",
            subcategory="x",
            scenario="x",
            target_label="a",
        )
        # Invalid task_type should cause worker to fail gracefully, returning []
        results = engine.generate_batch(
            nodes=[node],
            personas=[sample_persona],
            spec=classification_spec,
            task_type="not_a_real_type",  # type: ignore[arg-type]
            n=1,
        )
        assert results == []


# ---------------------------------------------------------------------------
# CritiqueEngine
# ---------------------------------------------------------------------------


class TestCritiqueEngine:
    @patch("tessera.pipeline.critique.get_client")
    def test_score_classification(
        self, mock_get_client, classification_spec
    ) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(
            {"realism": 8.0, "label_correctness": 7.5, "specificity": 9.0, "reasoning": "good"}
        )
        mock_get_client.return_value = mock_client

        engine = CritiqueEngine()
        ex = Example(
            task_type=TaskType.CLASSIFICATION,
            text="I lost my card",
            label="card_lost",
        )
        scores = engine.score(ex, classification_spec, TaskType.CLASSIFICATION)
        assert scores.mean == pytest.approx((8.0 + 7.5 + 9.0) / 3)

    @patch("tessera.pipeline.critique.get_client")
    def test_score_qa_maps_fields_correctly(
        self, mock_get_client, qa_spec
    ) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(
            {
                "groundedness": 9.0,
                "question_clarity": 8.0,
                "answer_completeness": 7.0,
                "reasoning": "ok",
            }
        )
        mock_get_client.return_value = mock_client

        engine = CritiqueEngine()
        ex = Example(
            task_type=TaskType.QA,
            context="Science text",
            question="What is photosynthesis?",
            answer="Process by which plants make food.",
        )
        scores = engine.score(ex, qa_spec, TaskType.QA)
        assert scores.realism == 9.0
        assert scores.label_correctness == 8.0
        assert scores.specificity == 7.0

    def test_wrong_spec_raises_configuration_error(self, extraction_spec) -> None:
        engine = CritiqueEngine()
        ex = Example(
            task_type=TaskType.CLASSIFICATION,
            text="text",
            label="lbl",
        )
        with pytest.raises(ConfigurationError):
            engine.score(ex, extraction_spec, TaskType.CLASSIFICATION)
