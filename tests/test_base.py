"""Tests for TaskTemplate base class — pipeline orchestration logic."""
from __future__ import annotations

import pytest

from tessera.core.base import TaskTemplate
from tessera.core.exceptions import ConfigurationError
from tessera.core.models import Example, TaskType, Taxonomy, TaxonomyNode

# ---------------------------------------------------------------------------
# _sample_nodes_balanced
# ---------------------------------------------------------------------------


class TestSampleNodesBalanced:
    def test_empty_taxonomy_returns_empty(self) -> None:
        taxonomy = Taxonomy(task_type=TaskType.CLASSIFICATION, nodes=[])
        result = TaskTemplate._sample_nodes_balanced(taxonomy, 10)
        assert result == []

    def test_returns_n_nodes(self, classification_taxonomy) -> None:
        result = TaskTemplate._sample_nodes_balanced(classification_taxonomy, 9)
        assert len(result) == 9

    def test_balanced_across_labels(self, classification_taxonomy) -> None:
        result = TaskTemplate._sample_nodes_balanced(classification_taxonomy, 9)
        labels = [n.target_label for n in result]
        counts = {lbl: labels.count(lbl) for lbl in set(labels)}
        for count in counts.values():
            assert count == 3

    def test_single_label_still_fills_quota(self) -> None:
        taxonomy = Taxonomy(
            task_type=TaskType.CLASSIFICATION,
            nodes=[
                TaxonomyNode(
                    label="only",
                    category="x",
                    subcategory="x",
                    scenario="x",
                    target_label="only",
                )
            ],
        )
        result = TaskTemplate._sample_nodes_balanced(taxonomy, 5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _trim_to_balance
# ---------------------------------------------------------------------------


class TestTrimToBalance:
    def test_empty_returns_empty(self) -> None:
        assert TaskTemplate._trim_to_balance([], 10) == []

    def test_trims_to_n_with_balance(self) -> None:
        examples = [
            Example(task_type=TaskType.CLASSIFICATION, text=f"text_{i}", label=lbl)
            for lbl in ["a", "b", "c"]
            for i in range(10)
        ]
        result = TaskTemplate._trim_to_balance(examples, 9)
        assert len(result) == 9
        labels = [ex.label for ex in result]
        for lbl in ["a", "b", "c"]:
            assert labels.count(lbl) == 3

    def test_non_labelled_examples_head_trim(self) -> None:
        examples = [
            Example(
                task_type=TaskType.INSTRUCTION,
                instruction=f"instr_{i}",
                response=f"resp_{i}",
            )
            for i in range(20)
        ]
        result = TaskTemplate._trim_to_balance(examples, 5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# run_pipeline input validation
# ---------------------------------------------------------------------------


class TestRunPipelineValidation:
    def test_rejects_zero_n_examples(
        self, classification_spec, sample_persona
    ) -> None:
        from tessera.tasks.classification import ClassificationTask

        task = ClassificationTask()
        with pytest.raises(ConfigurationError, match="n_examples must be > 0"):
            task.run_pipeline(
                spec=classification_spec,
                personas=[sample_persona],
                n_examples=0,
            )

    def test_rejects_negative_n_examples(
        self, classification_spec, sample_persona
    ) -> None:
        from tessera.tasks.classification import ClassificationTask

        task = ClassificationTask()
        with pytest.raises(ConfigurationError, match="n_examples must be > 0"):
            task.run_pipeline(
                spec=classification_spec,
                personas=[sample_persona],
                n_examples=-5,
            )

    def test_rejects_empty_personas(self, classification_spec) -> None:
        from tessera.tasks.classification import ClassificationTask

        task = ClassificationTask()
        with pytest.raises(ConfigurationError, match="personas"):
            task.run_pipeline(
                spec=classification_spec,
                personas=[],
                n_examples=10,
            )

    def test_rejects_negative_max_retries(
        self, classification_spec, sample_persona
    ) -> None:
        from tessera.tasks.classification import ClassificationTask

        task = ClassificationTask()
        with pytest.raises(ConfigurationError, match="max_retries"):
            task.run_pipeline(
                spec=classification_spec,
                personas=[sample_persona],
                n_examples=10,
                max_retries=-1,
            )
