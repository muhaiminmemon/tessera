"""Pydantic v2 domain models shared across the entire Tessera pipeline."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    INSTRUCTION = "instruction"


# ---------------------------------------------------------------------------
# Task specification models
# ---------------------------------------------------------------------------


class ClassificationSpec(BaseModel):
    domain: str
    labels: list[str] = Field(min_length=2)
    label_descriptions: dict[str, str] = Field(default_factory=dict)
    language: str = "English"
    example_inputs: list[str] = Field(default_factory=list)

    @field_validator("labels")
    @classmethod
    def labels_unique_and_sufficient(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("at least 2 labels required")
        if len(v) != len(set(v)):
            raise ValueError("labels must be unique")
        return v


class ExtractionSpec(BaseModel):
    domain: str
    schema_definition: dict[str, str]
    source_text_type: str = "document"
    language: str = "English"


class InstructionSpec(BaseModel):
    domain: str
    instruction_types: list[str]
    response_format: str = "prose"
    language: str = "English"


TaskSpec = ClassificationSpec | ExtractionSpec | InstructionSpec


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TaxonomyNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    category: str
    subcategory: str
    scenario: str
    depth: int = 1
    target_label: str


class Taxonomy(BaseModel):
    task_type: TaskType
    nodes: list[TaxonomyNode] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def nodes_for_label(self, label: str) -> list[TaxonomyNode]:
        return [n for n in self.nodes if n.target_label == label]

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


class Persona(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    formality: str  # formal | semi-formal | casual | very_casual
    expertise: str  # expert | intermediate | novice | none
    age_range: str
    cultural_context: str
    writing_quirks: list[str] = Field(default_factory=list)

    def to_prompt_fragment(self) -> str:
        quirks = ", ".join(self.writing_quirks) if self.writing_quirks else "none"
        return (
            f"Persona: {self.name} | Formality: {self.formality} | "
            f"Expertise: {self.expertise} | Age: {self.age_range} | "
            f"Cultural context: {self.cultural_context} | Writing quirks: {quirks}"
        )


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------


class CritiqueScores(BaseModel):
    realism: float = Field(ge=0, le=10)
    label_correctness: float = Field(ge=0, le=10)
    specificity: float = Field(ge=0, le=10)
    reasoning: str = ""

    @property
    def mean(self) -> float:
        return (self.realism + self.label_correctness + self.specificity) / 3

    def passes(self, threshold: float) -> bool:
        return self.mean >= threshold


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType

    # Classification
    text: Optional[str] = None
    label: Optional[str] = None

    # Extraction
    source_text: Optional[str] = None
    extracted_fields: Optional[dict[str, Any]] = None

    # Instruction
    instruction: Optional[str] = None
    response: Optional[str] = None

    # Metadata
    taxonomy_node_id: Optional[str] = None
    persona_id: Optional[str] = None
    model_used: Optional[str] = None
    critique_scores: Optional[CritiqueScores] = None
    passed_critique: bool = False
    embedding_id: Optional[str] = None

    @model_validator(mode="after")
    def check_required_fields_per_task_type(self) -> "Example":
        if self.task_type == TaskType.CLASSIFICATION:
            if self.text is None:
                raise ValueError("text is required for CLASSIFICATION examples")
            if self.label is None:
                raise ValueError("label is required for CLASSIFICATION examples")
        elif self.task_type == TaskType.EXTRACTION:
            if self.source_text is None:
                raise ValueError("source_text is required for EXTRACTION examples")
            if self.extracted_fields is None:
                raise ValueError("extracted_fields is required for EXTRACTION examples")
        elif self.task_type == TaskType.INSTRUCTION:
            if self.instruction is None:
                raise ValueError("instruction is required for INSTRUCTION examples")
            if self.response is None:
                raise ValueError("response is required for INSTRUCTION examples")
        return self


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------


class GenerationResult(BaseModel):
    task_type: TaskType
    spec: dict[str, Any]
    examples: list[Example]
    total_generated: int
    total_after_critique: int
    total_after_dedup: int
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationMetrics(BaseModel):
    task_type: TaskType
    model: str
    dataset: str
    n_train: int
    n_test: int
    f1_macro: float = 0.0
    accuracy: float = 0.0
    per_field_f1: dict[str, float] = Field(default_factory=dict)
    json_validity_rate: float = 0.0
    llm_judge_score: float = 0.0
    real_data_f1: float = 0.0
    random_baseline_f1: float = 0.0

    @property
    def pct_of_real_data(self) -> float:
        if self.real_data_f1 == 0:
            return 0.0
        return (self.f1_macro / self.real_data_f1) * 100
