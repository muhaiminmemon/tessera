"""Shared pytest fixtures for Tessera tests."""
from __future__ import annotations

import pytest

from tessera.core.models import (
    ClassificationSpec,
    ExtractionSpec,
    InstructionSpec,
    Persona,
    QASpec,
    Taxonomy,
    TaxonomyNode,
)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_spec() -> ClassificationSpec:
    return ClassificationSpec(
        domain="banking customer support",
        labels=["account_locked", "balance_inquiry", "card_lost"],
    )


@pytest.fixture()
def extraction_spec() -> ExtractionSpec:
    return ExtractionSpec(
        domain="legal contracts",
        schema_definition={"party_a": "str", "party_b": "str", "effective_date": "str"},
    )


@pytest.fixture()
def instruction_spec() -> InstructionSpec:
    return InstructionSpec(
        domain="Python programming",
        instruction_types=["explain", "write_code", "debug"],
    )


@pytest.fixture()
def qa_spec() -> QASpec:
    return QASpec(
        domain="science encyclopedia articles",
        question_types=["factoid", "multi-hop", "abstractive", "unanswerable"],
    )


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_persona() -> Persona:
    return Persona(
        name="Test User",
        formality="formal",
        expertise="expert",
        age_range="30-40",
        cultural_context="North American",
        writing_quirks=["concise"],
    )


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_taxonomy(classification_spec: ClassificationSpec) -> Taxonomy:
    from tessera.core.models import TaskType

    nodes = [
        TaxonomyNode(
            label=lbl,
            category="banking",
            subcategory=lbl,
            scenario=f"A customer asks about {lbl}",
            target_label=lbl,
        )
        for lbl in classification_spec.labels
    ]
    return Taxonomy(task_type=TaskType.CLASSIFICATION, nodes=nodes)


@pytest.fixture()
def qa_taxonomy(qa_spec: QASpec) -> Taxonomy:
    from tessera.core.models import TaskType

    nodes = [
        TaxonomyNode(
            label=qt,
            category="science",
            subcategory=qt,
            scenario=f"A {qt} question about science",
            target_label=qt,
        )
        for qt in qa_spec.question_types
    ]
    return Taxonomy(task_type=TaskType.QA, nodes=nodes)
