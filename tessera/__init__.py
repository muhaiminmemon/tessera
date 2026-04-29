"""
Tessera — multi-task synthetic data framework for fine-tuning small LLMs.

Quick start:
    from tessera import generate
    result = generate("classification", {"domain": "banking", "labels": ["spam", "ham"]}, n_examples=100)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from tessera.core.models import (
    ClassificationSpec,
    ExtractionSpec,
    GenerationResult,
    InstructionSpec,
    TaskType,
)
from tessera.core.personas import get_all_personas

__version__ = "0.1.0"
__all__ = ["generate", "__version__"]


def generate(
    task: str,
    spec_dict: dict[str, Any],
    n_examples: int = 1000,
    model: Optional[str] = None,
    critique_threshold: float = 6.0,
    output_format: str = "jsonl",
    output_path: Optional[str] = None,
) -> GenerationResult:
    """
    Generate a synthetic dataset end-to-end.

    Parameters
    ----------
    task:
        One of "classification", "extraction", "instruction".
    spec_dict:
        Dictionary matching the spec for the chosen task type.
    n_examples:
        Target number of examples in the final dataset.
    model:
        LLM model name. Defaults to TESSERA_DEFAULT_MODEL env var or gpt-4o-mini.
    critique_threshold:
        Minimum mean critique score (0-10) for an example to pass.
    output_format:
        "jsonl", "alpaca", or "sharegpt".
    output_path:
        If set, writes the formatted dataset to this file.

    Returns
    -------
    GenerationResult
    """
    from tessera.tasks.classification import ClassificationTask
    from tessera.tasks.extraction import ExtractionTask
    from tessera.tasks.instruction import InstructionTask

    model = model or os.environ.get("TESSERA_DEFAULT_MODEL", "gpt-4o-mini")
    task_type = TaskType(task.lower())
    personas = get_all_personas()

    if task_type == TaskType.CLASSIFICATION:
        spec = ClassificationSpec(**spec_dict)
        task_obj = ClassificationTask(model=model, critique_model=model)
    elif task_type == TaskType.EXTRACTION:
        spec = ExtractionSpec(**spec_dict)
        task_obj = ExtractionTask(model=model, critique_model=model)
    elif task_type == TaskType.INSTRUCTION:
        spec = InstructionSpec(**spec_dict)
        task_obj = InstructionTask(model=model, critique_model=model)
    else:
        raise ValueError(f"Unknown task type: {task!r}. Choose from: classification, extraction, instruction")

    result = task_obj.run_pipeline(
        spec=spec,
        personas=personas,
        n_examples=n_examples,
        critique_threshold=critique_threshold,
    )

    if output_path:
        formatted = task_obj.format_for_finetuning(result.examples, fmt=output_format)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for row in formatted:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return result
