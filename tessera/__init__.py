"""
Tessera — multi-task synthetic data framework for fine-tuning small LLMs.

Quick start:
    from tessera import generate
    result = generate("classification", {"domain": "banking", "labels": ["spam", "ham"]}, n_examples=100)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

from dotenv import load_dotenv

# Always load .env from the repo root, regardless of where Python is invoked from.
load_dotenv(Path(__file__).parent.parent / ".env")

from tessera.core.exceptions import (
    ConfigurationError,
    CritiqueError,
    DeduplicationError,
    GenerationError,
    TaxonomyError,
    TesseraError,
)
from tessera.core.models import (
    ClassificationSpec,
    ExtractionSpec,
    GenerationResult,
    InstructionSpec,
    QASpec,
    TaskType,
)
from tessera.core.personas import get_all_personas
from tessera.tasks.qa import QATask

__version__ = "0.1.0"
__all__ = [
    "generate",
    "QATask",
    "__version__",
    # Exceptions
    "TesseraError",
    "ConfigurationError",
    "GenerationError",
    "CritiqueError",
    "TaxonomyError",
    "DeduplicationError",
]


def generate(
    task: str,
    spec_dict: dict[str, Any],
    n_examples: int = 1000,
    model: Optional[str] = None,
    critique_threshold: Optional[float] = None,
    output_format: str = "jsonl",
    output_path: Optional[str] = None,
    max_retries: int = 3,
) -> GenerationResult:
    """
    Generate a synthetic dataset end-to-end.

    Parameters
    ----------
    task:
        One of "classification", "extraction", "instruction", "qa".
    spec_dict:
        Dictionary matching the spec for the chosen task type.
    n_examples:
        Target number of examples in the final dataset.  The pipeline retries
        up to ``max_retries`` times to guarantee this count is reached.
    model:
        LLM model name. Defaults to TESSERA_DEFAULT_MODEL env var or gpt-4o-mini.
    critique_threshold:
        Minimum mean critique score (0-10) for an example to pass.
        Defaults to None, which uses each task's built-in default
        (classification: 7.0, extraction: 7.5, instruction: 7.0, qa: 7.5).
    output_format:
        "jsonl", "alpaca", or "sharegpt".
    output_path:
        If set, writes the formatted dataset to this file.
    max_retries:
        How many additional generate→critique→dedup passes to run if the
        pool falls short of n_examples after the initial pass. Default 3.
        Set to 0 to disable retries (returns whatever the first pass yields).

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
    elif task_type == TaskType.QA:
        spec = QASpec(**spec_dict)
        task_obj = QATask(model=model, critique_model=model)
    else:
        raise ValueError(
            f"Unknown task type: {task!r}. "
            "Choose from: classification, extraction, instruction, qa"
        )

    t0 = time.time()
    result = task_obj.run_pipeline(
        spec=spec,
        personas=personas,
        n_examples=n_examples,
        critique_threshold=critique_threshold,
        max_retries=max_retries,
    )
    elapsed = time.time() - t0
    log.info(
        "done: %d examples | cost $%.4f | %.1fs",
        len(result.examples), result.cost_usd, elapsed,
    )

    if output_path:
        formatted = task_obj.format_for_finetuning(result.examples, fmt=output_format)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in formatted:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return result
