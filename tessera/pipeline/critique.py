"""Critique engine: scores each Example on realism, correctness, specificity."""
from __future__ import annotations

import json

from tessera.core.llm_client import get_client
from tessera.core.models import (
    CritiqueScores,
    Example,
    TaskSpec,
    TaskType,
    ClassificationSpec,
    ExtractionSpec,
    InstructionSpec,
)
from tessera.core import prompts


class CritiqueEngine:
    def score(
        self,
        example: Example,
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
    ) -> CritiqueScores:
        client = get_client()

        if task_type == TaskType.CLASSIFICATION:
            assert isinstance(spec, ClassificationSpec)
            sys_msg = prompts.classification_critique_system(example, spec)
            usr_msg = prompts.classification_critique_user(example, spec)
        elif task_type == TaskType.EXTRACTION:
            assert isinstance(spec, ExtractionSpec)
            sys_msg = prompts.extraction_critique_system(example, spec)
            usr_msg = prompts.extraction_critique_user(example, spec)
        elif task_type == TaskType.INSTRUCTION:
            assert isinstance(spec, InstructionSpec)
            sys_msg = prompts.instruction_critique_system(example, spec)
            usr_msg = prompts.instruction_critique_user(example, spec)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        raw = client.complete(
            model=model,
            system=sys_msg,
            user=usr_msg,
            temperature=0.2,
            max_tokens=512,
            json_mode=True,
        )

        data = json.loads(raw)

        return CritiqueScores(
            realism=float(data.get("realism", 0)),
            label_correctness=float(data.get("label_correctness", 0)),
            specificity=float(data.get("specificity", 0)),
            reasoning=str(data.get("reasoning", "")),
        )
