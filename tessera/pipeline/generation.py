"""Batch generation engine: turns taxonomy nodes + personas into raw Examples."""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tessera.core.llm_client import get_client
from tessera.core.models import (
    Example,
    Persona,
    TaskSpec,
    TaskType,
    TaxonomyNode,
    ClassificationSpec,
    ExtractionSpec,
    InstructionSpec,
    QASpec,
)
from tessera.core import prompts


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        inner = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(inner).strip()
    return json.loads(text)


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
        max_workers = int(os.environ.get("TESSERA_MAX_CONCURRENT", "10"))

        def _worker(node: TaxonomyNode) -> Example | None:
            persona = random.choice(personas)
            try:
                return self._generate_one(client, node, persona, spec, task_type, model)
            except Exception as exc:
                print(f"[GenerationEngine] failed node={node.id} persona={persona.name}: {exc}")
                return None

        examples: list[Example] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, node): node for node in sample}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    examples.append(result)

        return examples

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

        assert isinstance(client, LLMClient)

        # QA makes two sequential LLM calls (context then QA pair) so it is
        # handled as a full early-return before the shared single-call path.
        if task_type == TaskType.QA:
            assert isinstance(spec, QASpec)
            ctx_raw = client.complete(
                model=model,
                system=prompts.qa_context_generation_system(node, persona, spec),
                user=prompts.qa_context_generation_user(node, persona, spec),
                temperature=0.9,
                max_tokens=600,
                json_mode=True,
            )
            context = _parse_json(ctx_raw).get("context", "").strip()
            if not context:
                raise ValueError("LLM returned empty context field")
            question_type = node.target_label
            qa_raw = client.complete(
                model=model,
                system=prompts.qa_pair_generation_system(spec),
                user=prompts.qa_pair_generation_user(context, question_type),
                temperature=0.7,
                max_tokens=400,
                json_mode=True,
            )
            qa_data = _parse_json(qa_raw)
            question = qa_data.get("question", "").strip()
            answer = qa_data.get("answer", "").strip()
            if not question or not answer:
                raise ValueError("LLM returned empty question or answer")
            difficulty = qa_data.get("difficulty", "medium").strip()
            return Example(
                task_type=task_type,
                context=context,
                question=question,
                answer=answer,
                question_type=question_type,
                difficulty=difficulty,
                label=question_type,
                taxonomy_node_id=node.id,
                persona_id=persona.id,
                model_used=model,
            )

        if task_type == TaskType.CLASSIFICATION:
            assert isinstance(spec, ClassificationSpec)
            sys_msg = prompts.classification_generation_system(node, persona, spec)
            usr_msg = prompts.classification_generation_user(node, persona, spec)
        elif task_type == TaskType.EXTRACTION:
            assert isinstance(spec, ExtractionSpec)
            sys_msg = prompts.extraction_generation_system(node, persona, spec)
            usr_msg = prompts.extraction_generation_user(node, persona, spec)
        elif task_type == TaskType.INSTRUCTION:
            assert isinstance(spec, InstructionSpec)
            sys_msg = prompts.instruction_generation_system(node, persona, spec)
            usr_msg = prompts.instruction_generation_user(node, persona, spec)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        raw = client.complete(
            model=model,
            system=sys_msg,
            user=usr_msg,
            temperature=0.9,
            max_tokens=1024,
            json_mode=True,
        )

        data = _parse_json(raw)

        if task_type == TaskType.CLASSIFICATION:
            assert isinstance(spec, ClassificationSpec)
            label = data.get("label", "")
            # Fall back to the node's target_label rather than discarding the example.
            if label not in spec.labels:
                label = node.target_label
            text = data.get("text", "").strip()
            if not text:
                raise ValueError("LLM returned empty text field")
            return Example(
                task_type=task_type,
                text=text,
                label=label,
                taxonomy_node_id=node.id,
                persona_id=persona.id,
                model_used=model,
            )
        elif task_type == TaskType.EXTRACTION:
            extracted = data.get("extracted_fields", {})
            if isinstance(extracted, list):
                extracted = extracted[0] if extracted else {}
            return Example(
                task_type=task_type,
                source_text=data["source_text"],
                extracted_fields=extracted,
                taxonomy_node_id=node.id,
                persona_id=persona.id,
                model_used=model,
            )
        else:  # INSTRUCTION
            return Example(
                task_type=task_type,
                instruction=data["instruction"],
                response=data["response"],
                taxonomy_node_id=node.id,
                persona_id=persona.id,
                model_used=model,
            )
