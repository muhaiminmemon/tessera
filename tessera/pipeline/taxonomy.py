"""Expands a task spec into a Taxonomy of generation-ready nodes."""
from __future__ import annotations

import json
import warnings

from tessera.core.llm_client import get_client
from tessera.core.models import (
    ClassificationSpec,
    ExtractionSpec,
    InstructionSpec,
    TaskSpec,
    TaskType,
    Taxonomy,
    TaxonomyNode,
)
from tessera.core import prompts


class TaxonomyExpander:
    def expand(
        self,
        spec: TaskSpec,
        task_type: TaskType,
        model: str = "gpt-4o-mini",
    ) -> Taxonomy:
        client = get_client()

        if task_type == TaskType.CLASSIFICATION:
            assert isinstance(spec, ClassificationSpec)
            sys_msg = prompts.classification_taxonomy_system(spec)
            usr_msg = prompts.classification_taxonomy_user(spec)
        elif task_type == TaskType.EXTRACTION:
            assert isinstance(spec, ExtractionSpec)
            sys_msg = prompts.extraction_taxonomy_system(spec)
            usr_msg = prompts.extraction_taxonomy_user(spec)
        elif task_type == TaskType.INSTRUCTION:
            assert isinstance(spec, InstructionSpec)
            sys_msg = prompts.instruction_taxonomy_system(spec)
            usr_msg = prompts.instruction_taxonomy_user(spec)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        raw = client.complete(
            model=model,
            system=sys_msg,
            user=usr_msg,
            temperature=0.7,
            max_tokens=4096,
            json_mode=True,
        )

        data = json.loads(raw)
        nodes_data = data.get("nodes", [])

        nodes: list[TaxonomyNode] = []
        for nd in nodes_data:
            try:
                nodes.append(
                    TaxonomyNode(
                        label=nd.get("label", ""),
                        category=nd.get("category", ""),
                        subcategory=nd.get("subcategory", ""),
                        scenario=nd.get("scenario", ""),
                        depth=int(nd.get("depth", 1)),
                        target_label=nd.get("target_label", ""),
                    )
                )
            except Exception as exc:
                warnings.warn(f"[TaxonomyExpander] skipped malformed node: {exc}")

        taxonomy = Taxonomy(task_type=task_type, nodes=nodes)

        # Validate coverage
        if task_type == TaskType.CLASSIFICATION:
            assert isinstance(spec, ClassificationSpec)
            for label in spec.labels:
                if not taxonomy.nodes_for_label(label):
                    warnings.warn(
                        f"[TaxonomyExpander] label '{label}' has no taxonomy nodes. "
                        "Consider re-running or increasing max_tokens."
                    )
        elif task_type == TaskType.INSTRUCTION:
            assert isinstance(spec, InstructionSpec)
            for itype in spec.instruction_types:
                if not taxonomy.nodes_for_label(itype):
                    warnings.warn(
                        f"[TaxonomyExpander] instruction_type '{itype}' has no taxonomy nodes."
                    )

        return taxonomy
