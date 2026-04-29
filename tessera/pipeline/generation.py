"""Batch generation engine: turns taxonomy nodes + personas into raw Examples."""
from __future__ import annotations

import json
import random

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
        """Generate up to n examples by iterating over node/persona pairs."""
        client = get_client()
        target = n if n is not None else len(nodes)
        sample = nodes[:target]

        examples: list[Example] = []
        for node in sample:
            persona = random.choice(personas)
            try:
                ex = self._generate_one(client, node, persona, spec, task_type, model)
                examples.append(ex)
            except Exception as exc:
                print(f"[GenerationEngine] failed node={node.id} persona={persona.name}: {exc}")

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
            # The node always carries a valid spec label, so this is safe.
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
            return Example(
                task_type=task_type,
                source_text=data["source_text"],
                extracted_fields=data["extracted_fields"],
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
