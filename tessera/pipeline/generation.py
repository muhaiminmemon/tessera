"""Batch generation engine: turns taxonomy nodes + personas into raw Examples."""
from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from tessera.core import config as _cfg
from tessera.core import prompts
from tessera.core.exceptions import GenerationError
from tessera.core.llm_client import get_client
from tessera.core.models import (
    ClassificationSpec,
    Example,
    Persona,
    QASpec,
    TaskSpec,
    TaskType,
    TaxonomyNode,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dispatch table for prompt builders (system_fn, user_fn).
# QA is excluded because it requires two sequential LLM calls.
# ---------------------------------------------------------------------------

_PromptPair = tuple[Callable[..., str], Callable[..., str]]

_GEN_PROMPTS: dict[TaskType, _PromptPair] = {
    TaskType.CLASSIFICATION: (
        prompts.classification_generation_system,
        prompts.classification_generation_user,
    ),
    TaskType.EXTRACTION: (
        prompts.extraction_generation_system,
        prompts.extraction_generation_user,
    ),
    TaskType.INSTRUCTION: (
        prompts.instruction_generation_system,
        prompts.instruction_generation_user,
    ),
}


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [ln for ln in lines[1:] if ln.strip() != "```"]
        text = "\n".join(inner).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"LLM returned invalid JSON: {exc}. Raw response was: {raw!r}"
        ) from exc
    if not isinstance(data, dict):
        raise GenerationError(
            f"LLM returned JSON of type {type(data).__name__}, expected object. "
            f"Raw response was: {raw!r}"
        )
    return data


class GenerationEngine:
    def generate_batch(
        self,
        nodes: list[TaxonomyNode],
        personas: list[Persona],
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
        n: int | None = None,
    ) -> list[Example]:
        """Generate examples in parallel using ThreadPoolExecutor."""
        client = get_client()
        target = n if n is not None else len(nodes)
        sample = nodes[:target]
        max_workers = _cfg.max_concurrent()

        def _worker(node: TaxonomyNode) -> Example | None:
            persona = random.choice(personas)
            try:
                return self._generate_one(client, node, persona, spec, task_type, model)
            except Exception as exc:
                log.warning("generation failed node=%s persona=%s: %s", node.id, persona.name, exc)
                return None

        examples: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, node): node for node in sample}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    examples.append(result)

        return examples

    def generate_one(
        self,
        node: TaxonomyNode,
        persona: Persona,
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
    ) -> Example:
        """Generate a single example synchronously, without spawning a thread pool.

        Use this from within an already-parallelised context (e.g. TaskTemplate
        worker threads) to avoid creating a nested ThreadPoolExecutor.
        """
        client = get_client()
        return self._generate_one(client, node, persona, spec, task_type, model)

    def _generate_one(
        self,
        client: object,
        node: TaxonomyNode,
        persona: Persona,
        spec: TaskSpec,
        task_type: TaskType,
        model: str,
    ) -> Example:
        from tessera.core.llm_client import LLMClient

        if not isinstance(client, LLMClient):
            raise GenerationError(f"Expected LLMClient, got {type(client).__name__}")

        # QA requires two sequential calls — handle before the shared single-call path.
        if task_type == TaskType.QA:
            return self._generate_qa(client, node, persona, spec, model)

        if task_type not in _GEN_PROMPTS:
            raise GenerationError(f"Unknown task_type: {task_type}")

        sys_fn, usr_fn = _GEN_PROMPTS[task_type]
        raw = client.complete(
            model=model,
            system=sys_fn(node, persona, spec),
            user=usr_fn(node, persona, spec),
            temperature=_cfg.GENERATION_TEMPERATURE,
            max_tokens=1024,
            json_mode=True,
        )

        data = _parse_json(raw)
        return self._build_example(task_type, data, node, persona, spec, model)

    # ------------------------------------------------------------------
    # Per-task example construction helpers
    # ------------------------------------------------------------------

    def _generate_qa(
        self,
        client: object,
        node: TaxonomyNode,
        persona: Persona,
        spec: TaskSpec,
        model: str,
    ) -> Example:
        from tessera.core.llm_client import LLMClient

        if not isinstance(spec, QASpec):
            raise GenerationError(f"Expected QASpec for QA task, got {type(spec).__name__}")
        if not isinstance(client, LLMClient):
            raise GenerationError(f"Expected LLMClient, got {type(client).__name__}")

        ctx_raw = client.complete(
            model=model,
            system=prompts.qa_context_generation_system(node, persona, spec),
            user=prompts.qa_context_generation_user(node, persona, spec),
            temperature=_cfg.CONTEXT_GENERATION_TEMPERATURE,
            max_tokens=600,
            json_mode=True,
        )
        context = _parse_json(ctx_raw).get("context", "").strip()
        if not context:
            raise GenerationError("LLM returned empty context field")

        question_type = node.target_label
        qa_raw = client.complete(
            model=model,
            system=prompts.qa_pair_generation_system(spec),
            user=prompts.qa_pair_generation_user(context, question_type),
            temperature=_cfg.QA_PAIR_TEMPERATURE,
            max_tokens=400,
            json_mode=True,
        )
        qa_data = _parse_json(qa_raw)
        question = qa_data.get("question", "").strip()
        answer = qa_data.get("answer", "").strip()
        if not question or not answer:
            raise GenerationError("LLM returned empty question or answer")

        return Example(
            task_type=TaskType.QA,
            context=context,
            question=question,
            answer=answer,
            question_type=question_type,
            difficulty=qa_data.get("difficulty", "medium").strip(),
            label=question_type,
            taxonomy_node_id=node.id,
            persona_id=persona.id,
            model_used=model,
        )

    @staticmethod
    def _build_example(
        task_type: TaskType,
        data: dict,
        node: TaxonomyNode,
        persona: Persona,
        spec: TaskSpec,
        model: str,
    ) -> Example:
        """Construct an Example from parsed LLM JSON for non-QA task types."""
        common: dict[str, Any] = dict(
            task_type=task_type,
            taxonomy_node_id=node.id,
            persona_id=persona.id,
            model_used=model,
        )

        if task_type == TaskType.CLASSIFICATION:
            if not isinstance(spec, ClassificationSpec):
                raise GenerationError(
                    f"Expected ClassificationSpec, got {type(spec).__name__}"
                )
            text = data.get("text", "").strip()
            if not text:
                raise GenerationError("LLM returned empty text field")
            label = data.get("label", "")
            if label not in spec.labels:
                label = node.target_label
            return Example(**common, text=text, label=label)

        if task_type == TaskType.EXTRACTION:
            extracted = data.get("extracted_fields", {})
            if isinstance(extracted, list):
                extracted = extracted[0] if extracted else {}
            return Example(
                **common,
                source_text=data["source_text"],
                extracted_fields=extracted,
            )

        # INSTRUCTION
        return Example(
            **common,
            instruction=data["instruction"],
            response=data["response"],
        )
