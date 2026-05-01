"""Critique engine: scores each Example on realism, correctness, specificity."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tessera.core.llm_client import get_client
from tessera.core.models import (
    CritiqueScores,
    Example,
    TaskSpec,
    TaskType,
    ClassificationSpec,
    ExtractionSpec,
    InstructionSpec,
    QASpec,
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
        elif task_type == TaskType.QA:
            assert isinstance(spec, QASpec)
            sys_msg = prompts.qa_critique_system(example, spec)
            usr_msg = prompts.qa_critique_user(example, spec)
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

        # QA critique returns groundedness/question_clarity/answer_completeness;
        # map onto the three standard CritiqueScores axes.
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

    def score_batch(
        self,
        examples: list[Example],
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
        threshold: float = 6.0,
    ) -> list[Example]:
        """Score a batch of examples in parallel; warn if 100% pass rate."""
        max_workers = int(os.environ.get("TESSERA_MAX_CONCURRENT", "10"))

        def _worker(ex: Example) -> Example | None:
            try:
                scores = self.score(ex, spec, task_type, model)
                ex.critique_scores = scores
                ex.passed_critique = scores.passes(threshold)
                return ex
            except Exception as exc:
                print(f"[CritiqueEngine] failed for example {ex.id}: {exc}")
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
                print(
                    "[tessera] warning: 100% critique pass rate — "
                    "consider raising threshold to 7.0"
                )

        return scored
