"""Critique engine: scores each Example on realism, correctness, specificity."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from tessera.core import config as _cfg
from tessera.core import prompts
from tessera.core.exceptions import ConfigurationError, CritiqueError
from tessera.core.llm_client import get_client
from tessera.core.models import (
    ClassificationSpec,
    CritiqueScores,
    Example,
    ExtractionSpec,
    InstructionSpec,
    QASpec,
    TaskSpec,
    TaskType,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dispatch table for prompt builders — extend here when adding a task type.
# ---------------------------------------------------------------------------

_PromptPair = tuple[Callable[..., str], Callable[..., str]]

_CRITIQUE_PROMPTS: dict[TaskType, _PromptPair] = {
    TaskType.CLASSIFICATION: (
        prompts.classification_critique_system,
        prompts.classification_critique_user,
    ),
    TaskType.EXTRACTION: (
        prompts.extraction_critique_system,
        prompts.extraction_critique_user,
    ),
    TaskType.INSTRUCTION: (
        prompts.instruction_critique_system,
        prompts.instruction_critique_user,
    ),
    TaskType.QA: (
        prompts.qa_critique_system,
        prompts.qa_critique_user,
    ),
}

# Expected spec type per task — used for type-safe validation.
_EXPECTED_SPEC: dict[TaskType, type] = {
    TaskType.CLASSIFICATION: ClassificationSpec,
    TaskType.EXTRACTION: ExtractionSpec,
    TaskType.INSTRUCTION: InstructionSpec,
    TaskType.QA: QASpec,
}


def _parse_scores(task_type: TaskType, data: dict) -> CritiqueScores:
    """Map raw LLM JSON to CritiqueScores, handling QA's different field names."""
    if task_type == TaskType.QA:
        return CritiqueScores(
            realism=float(data.get("groundedness", 0)),
            label_correctness=float(data.get("question_clarity", 0)),
            specificity=float(data.get("answer_completeness", 0)),
            reasoning=str(data.get("reasoning", "")),
        )
    return CritiqueScores(
        realism=float(data.get("realism", 0)),
        label_correctness=float(data.get("label_correctness", 0)),
        specificity=float(data.get("specificity", 0)),
        reasoning=str(data.get("reasoning", "")),
    )


class CritiqueEngine:
    def score(
        self,
        example: Example,
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
    ) -> CritiqueScores:
        expected = _EXPECTED_SPEC.get(task_type)
        if expected is None:
            raise ConfigurationError(f"Unknown task_type: {task_type}")
        if not isinstance(spec, expected):
            raise ConfigurationError(
                f"Expected {expected.__name__} for {task_type}, "
                f"got {type(spec).__name__}"
            )

        sys_fn, usr_fn = _CRITIQUE_PROMPTS[task_type]
        client = get_client()
        raw = client.complete(
            model=model,
            system=sys_fn(example, spec),
            user=usr_fn(example, spec),
            temperature=_cfg.CRITIQUE_TEMPERATURE,
            max_tokens=512,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CritiqueError(f"LLM returned invalid JSON for critique: {exc}") from exc

        return _parse_scores(task_type, data)

    def score_batch(
        self,
        examples: list[Example],
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
        threshold: float = 6.0,
    ) -> list[Example]:
        """Score a batch of examples in parallel; warn if 100% pass rate."""
        max_workers = _cfg.max_concurrent()

        def _worker(ex: Example) -> Example | None:
            try:
                scores = self.score(ex, spec, task_type, model)
                ex.critique_scores = scores
                ex.passed_critique = scores.passes(threshold)
                return ex
            except Exception as exc:
                log.warning("critique failed for example %s: %s", ex.id, exc)
                return None

        scored: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, ex): ex for ex in examples}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    scored.append(result)

        if scored:
            pass_rate = sum(1 for ex in scored if ex.passed_critique) / len(scored)
            if pass_rate >= 1.0 - 1e-9:
                log.warning(
                    "100%% critique pass rate — consider raising threshold to 7.0"
                )

        return scored
