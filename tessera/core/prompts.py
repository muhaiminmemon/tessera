"""Prompt builder functions for every pipeline stage across all three task types."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera.core.models import (
        ClassificationSpec,
        Example,
        ExtractionSpec,
        InstructionSpec,
        Persona,
        QASpec,
        TaxonomyNode,
    )


# ---------------------------------------------------------------------------
# Classification prompts
# ---------------------------------------------------------------------------


def classification_taxonomy_system(spec: "ClassificationSpec") -> str:
    return (
        "You are a taxonomy designer for a classification dataset. "
        "Your job is to produce a diverse set of realistic scenarios that "
        "would generate text belonging to each target label. "
        "Return ONLY valid JSON with no markdown fences."
    )


def classification_taxonomy_user(spec: "ClassificationSpec") -> str:
    label_desc = ""
    if spec.label_descriptions:
        label_desc = "\n".join(f"  - {k}: {v}" for k, v in spec.label_descriptions.items())
        label_desc = f"\nLabel descriptions:\n{label_desc}"

    return f"""Domain: {spec.domain}
Labels: {json.dumps(spec.labels)}{label_desc}
Language: {spec.language}

Generate a taxonomy of realistic scenarios for a text classification dataset.
Produce AT LEAST 3 nodes per label. Each node describes a specific situation
that would produce text belonging to that label.

Return JSON exactly like this:
{{
  "nodes": [
    {{
      "label": "<human-readable label>",
      "category": "<broad category>",
      "subcategory": "<narrow subcategory>",
      "scenario": "<concrete scenario description in 1-2 sentences>",
      "depth": 1,
      "target_label": "<one of {spec.labels}>"
    }}
  ]
}}

Vary styles, urgency levels, and contexts across nodes. Ensure every label
in {json.dumps(spec.labels)} has at least 3 nodes."""


def classification_generation_system(
    node: "TaxonomyNode", persona: "Persona", spec: "ClassificationSpec"
) -> str:
    return (
        f"You are a synthetic data generator. You write realistic user-authored text "
        f"messages, queries, or documents. "
        f"{persona.to_prompt_fragment()} "
        f"Write as this persona — match their vocabulary, formality, and typical errors. "
        f"Return ONLY valid JSON with no markdown fences."
    )


def classification_generation_user(
    node: "TaxonomyNode", persona: "Persona", spec: "ClassificationSpec"
) -> str:
    return f"""Task: Generate one realistic text example for a {spec.domain} classifier.

Scenario: {node.scenario}
Category: {node.category} / {node.subcategory}
Language: {spec.language}

You are the CUSTOMER writing to customer support. Never write as the support agent. Never include account balances, transaction amounts, or information a customer wouldn't know.
Persona: {persona.name} — {persona.formality} register, {persona.expertise} expertise. Match their vocabulary and typical phrasing.
Write 1-3 sentences maximum. No greetings, no sign-offs, no "Dear Support Team", no "Best regards". Just the core message.
Do NOT use emojis under any circumstances.
Do NOT include the label string in the text itself.

CRITICAL INSTRUCTION: You MUST return a JSON object with exactly two keys:
  "text"  — the generated example text (string)
  "label" — copy this string EXACTLY, character-for-character: {node.target_label!r}

The "label" value must be the exact string {node.target_label!r} — no other value is valid.

Return JSON (and nothing else):
{{"text": "<your generated text here>", "label": {json.dumps(node.target_label)}}}"""


def classification_critique_system(example: "Example", spec: "ClassificationSpec") -> str:
    return (
        "You are a quality evaluator for synthetic classification datasets. "
        "Score examples on three axes, each 0-10. Be strict and calibrated. "
        "Return ONLY valid JSON with no markdown fences."
    )


def classification_critique_user(example: "Example", spec: "ClassificationSpec") -> str:
    label_desc = spec.label_descriptions.get(example.label or "", "no description provided")
    return f"""Evaluate this synthetic classification example:

Domain: {spec.domain}
Text: {example.text}
Assigned label: {example.label}
Label description: {label_desc}
All labels: {json.dumps(spec.labels)}

Score each axis 0-10:
- realism: Does this text sound like something a real person would write? If the text reads like a bank employee or support agent responding TO a customer (rather than a customer writing TO support), set realism to 0.0 immediately — this is a generation error, not a valid example.
- label_correctness: Does the text clearly belong to this label and not others?
- specificity: Is the text concrete and scenario-specific (not generic)?

Return JSON:
{{
  "realism": <0-10>,
  "label_correctness": <0-10>,
  "specificity": <0-10>,
  "reasoning": "<1-2 sentence justification>"
}}"""


# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------


def extraction_taxonomy_system(spec: "ExtractionSpec") -> str:
    return (
        "You are a taxonomy designer for a structured extraction dataset. "
        "Generate diverse source-text scenarios that would allow an extractor "
        "to fill all fields in a given schema. "
        "Return ONLY valid JSON with no markdown fences."
    )


def extraction_taxonomy_user(spec: "ExtractionSpec") -> str:
    return f"""Domain: {spec.domain}
Source text type: {spec.source_text_type}
Schema to extract: {json.dumps(spec.schema_definition, indent=2)}
Language: {spec.language}

Generate a taxonomy of at least 12 diverse scenarios where {spec.source_text_type}
would contain all fields in the schema.

Return JSON:
{{
  "nodes": [
    {{
      "label": "<short label>",
      "category": "<broad category>",
      "subcategory": "<narrow subcategory>",
      "scenario": "<specific source text scenario, 1-2 sentences>",
      "depth": 1,
      "target_label": "<category>"
    }}
  ]
}}"""


def extraction_generation_system(
    node: "TaxonomyNode", persona: "Persona", spec: "ExtractionSpec"
) -> str:
    return (
        "You are a synthetic data generator for structured extraction tasks. "
        "Write a realistic source text and extract all schema fields from it. "
        f"{persona.to_prompt_fragment()} "
        "Return ONLY valid JSON with no markdown fences."
    )


def extraction_generation_user(
    node: "TaxonomyNode", persona: "Persona", spec: "ExtractionSpec"
) -> str:
    return f"""Generate a synthetic {spec.source_text_type} and extract all fields.

Domain: {spec.domain}
Scenario: {node.scenario}
Schema: {json.dumps(spec.schema_definition, indent=2)}
Language: {spec.language}

1. Write a realistic {spec.source_text_type} (50-300 words) that a persona like
   "{persona.name}" might produce or encounter.
2. Extract every schema field from the text.

Return extracted_fields as a single JSON object (dict), NOT a list or array. Extract exactly ONE relation triple.

Return JSON:
{{
  "source_text": "<the generated document>",
  "extracted_fields": {{
    {', '.join(f'"{k}": "<value or null>"' for k in spec.schema_definition)}
  }}
}}"""


def extraction_critique_system(example: "Example", spec: "ExtractionSpec") -> str:
    return (
        "You are a quality evaluator for synthetic extraction datasets. "
        "Score examples on three axes, each 0-10. "
        "Return ONLY valid JSON with no markdown fences."
    )


def extraction_critique_user(example: "Example", spec: "ExtractionSpec") -> str:
    return f"""Evaluate this synthetic extraction example:

Domain: {spec.domain}
Source text: {example.source_text}
Extracted fields: {json.dumps(example.extracted_fields, indent=2)}
Expected schema: {json.dumps(spec.schema_definition, indent=2)}

Score each axis 0-10:
- realism (axis stored as "realism"): Is the source text realistic and natural?
- label_correctness (axis stored as "label_correctness"): Are all schema fields
  present and accurately extracted from the source text?
- specificity (axis stored as "specificity"): Are extracted values specific
  and non-trivial (not empty strings or placeholders)?

Return JSON:
{{
  "realism": <0-10>,
  "label_correctness": <0-10>,
  "specificity": <0-10>,
  "reasoning": "<1-2 sentence justification>"
}}"""


# ---------------------------------------------------------------------------
# Instruction prompts
# ---------------------------------------------------------------------------


def instruction_taxonomy_system(spec: "InstructionSpec") -> str:
    return (
        "You are a taxonomy designer for an instruction-following dataset. "
        "Generate diverse task scenarios covering all instruction types. "
        "Return ONLY valid JSON with no markdown fences."
    )


def instruction_taxonomy_user(spec: "InstructionSpec") -> str:
    return f"""Domain: {spec.domain}
Instruction types: {json.dumps(spec.instruction_types)}
Response format: {spec.response_format}
Language: {spec.language}

Generate at least 3 nodes per instruction type describing specific tasks
a user might ask an AI assistant in the domain "{spec.domain}".

Return JSON:
{{
  "nodes": [
    {{
      "label": "<instruction type>",
      "category": "<broad category>",
      "subcategory": "<narrow subcategory>",
      "scenario": "<concrete task description, 1-2 sentences>",
      "depth": 1,
      "target_label": "<one of {spec.instruction_types}>"
    }}
  ]
}}"""


def instruction_generation_system(
    node: "TaxonomyNode", persona: "Persona", spec: "InstructionSpec"
) -> str:
    return (
        "You are a synthetic data generator for instruction-following datasets. "
        "Write a realistic instruction and a high-quality AI response. "
        f"{persona.to_prompt_fragment()} "
        "Write the instruction as this persona would phrase it. "
        "The response should be from a knowledgeable AI assistant. "
        "Return ONLY valid JSON with no markdown fences."
    )


def instruction_generation_user(
    node: "TaxonomyNode", persona: "Persona", spec: "InstructionSpec"
) -> str:
    return f"""Generate a synthetic instruction-response pair.

Domain: {spec.domain}
Instruction type: {node.target_label}
Scenario: {node.scenario}
Response format: {spec.response_format}
Language: {spec.language}

Write the instruction as persona "{persona.name}" would phrase it.
Write a complete, high-quality AI response.

Return JSON:
{{
  "instruction": "<the user instruction>",
  "response": "<the AI response>"
}}"""


def instruction_critique_system(example: "Example", spec: "InstructionSpec") -> str:
    return (
        "You are a quality evaluator for instruction-following datasets. "
        "Score examples on three axes, each 0-10. "
        "Return ONLY valid JSON with no markdown fences."
    )


def instruction_critique_user(example: "Example", spec: "InstructionSpec") -> str:
    return f"""Evaluate this instruction-response pair:

Domain: {spec.domain}
Instruction: {example.instruction}
Response: {example.response}

Score each axis 0-10:
- realism (axis "realism"): Does the instruction sound like something a real
  person would ask?
- label_correctness (axis "label_correctness"): Does the response fully and
  correctly address the instruction?
- specificity (axis "specificity"): Is the response specific, detailed, and
  genuinely helpful rather than generic?

Return JSON:
{{
  "realism": <0-10>,
  "label_correctness": <0-10>,
  "specificity": <0-10>,
  "reasoning": "<1-2 sentence justification>"
}}"""


# ---------------------------------------------------------------------------
# QA prompts
# ---------------------------------------------------------------------------


def qa_taxonomy_system(spec: "QASpec") -> str:
    return (
        "You are a taxonomy designer for a question-answering dataset. "
        "Generate diverse source-text scenarios pairing domain subtopics with question types. "
        "Return ONLY valid JSON with no markdown fences."
    )


def qa_taxonomy_user(spec: "QASpec") -> str:
    return f"""Domain: {spec.domain}
Question types: {json.dumps(spec.question_types)}
Language: {spec.language}

Generate at least 3 nodes per question type. Each node pairs a specific domain
subtopic with a question type, describing a document passage scenario that
naturally supports generating that question type.

Return JSON:
{{
  "nodes": [
    {{
      "label": "<subtopic + question type description>",
      "category": "<broad domain category>",
      "subcategory": "<specific subtopic>",
      "scenario": "<concrete document scenario in 1-2 sentences>",
      "depth": 1,
      "target_label": "<one of {spec.question_types}>"
    }}
  ]
}}

Ensure every question type in {json.dumps(spec.question_types)} has at least 3 nodes."""


def qa_context_generation_system(
    node: "TaxonomyNode", persona: "Persona", spec: "QASpec"
) -> str:
    return (
        "You are a synthetic document writer. Write realistic, factually plausible "
        "passages that resemble authentic documents from the given domain. "
        f"{persona.to_prompt_fragment()} "
        "Return ONLY valid JSON with no markdown fences."
    )


def qa_context_generation_user(
    node: "TaxonomyNode", persona: "Persona", spec: "QASpec"
) -> str:
    return f"""Write a realistic document passage for the following scenario.

Domain: {spec.domain}
Scenario: {node.scenario}
Category: {node.category} / {node.subcategory}
Question type this passage will support: {node.target_label}
Language: {spec.language}

Requirements:
- Write 150-300 words as a cohesive passage
- Include specific details: names, dates, numbers, or technical terms where appropriate
- Vary the structure: some passages use prose paragraphs, some include figures or structured data
- The passage must contain enough specific, attributable information to support
  a {node.target_label} question

Return JSON:
{{"context": "<the passage text>"}}"""


def qa_pair_generation_system(spec: "QASpec") -> str:
    return (
        "You are a QA dataset generator. Create high-quality question-answer pairs "
        "strictly grounded in a provided context passage. "
        "Return ONLY valid JSON with no markdown fences."
    )


_QA_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factoid": (
        "Ask about a single specific fact (name, date, number, or term) that is "
        "directly and explicitly stated in the passage. "
        "The answer must be a short phrase extracted verbatim or near-verbatim."
    ),
    "multi-hop": (
        "Ask a question that requires combining TWO separate pieces of information "
        "from different parts of the passage to arrive at the answer. "
        "The answer should synthesize both pieces into one coherent response."
    ),
    "abstractive": (
        "Ask a question whose answer requires paraphrasing, summarizing, or inferring "
        "from the passage — not direct extraction. "
        "The answer should demonstrate comprehension, not copy-paste."
    ),
    "unanswerable": (
        "Ask a question that looks relevant and plausible for this domain but "
        "CANNOT be answered from the passage alone — the information is simply absent. "
        'The answer MUST be exactly: "This cannot be determined from the provided context."'
    ),
}


def qa_pair_generation_user(context: str, question_type: str) -> str:
    instruction = _QA_TYPE_INSTRUCTIONS.get(
        question_type, _QA_TYPE_INSTRUCTIONS["factoid"]
    )
    return f"""Given the passage below, generate one {question_type} question-answer pair.

Context:
{context}

Question type: {question_type}
Instruction: {instruction}

Assess difficulty:
- easy: answer is obvious and directly stated
- medium: requires careful reading comprehension
- hard: requires careful reasoning or combining multiple details

Return JSON:
{{
  "question": "<the question>",
  "answer": "<the grounded answer>",
  "question_type": "{question_type}",
  "difficulty": "<easy | medium | hard>"
}}"""


def qa_critique_system(example: "Example", spec: "QASpec") -> str:
    return (
        "You are a quality evaluator for question-answering datasets. "
        "Score (context, question, answer) triples on four axes, each 0-10. Be strict. "
        "Return ONLY valid JSON with no markdown fences."
    )


def qa_critique_user(example: "Example", spec: "QASpec") -> str:
    return f"""Evaluate this QA example:

Domain: {spec.domain}
Context: {example.context}

Question: {example.question}
Answer: {example.answer}
Question type: {example.question_type}

Score each axis 0-10:
- groundedness: Is the answer actually supported by or derivable from the context?
  For "unanswerable" questions, the answer must be exactly
  "This cannot be determined from the provided context." — score 0 otherwise.
  Set to 0 if the answer contradicts or ignores the context.
- question_clarity: Is the question unambiguous, well-formed, and answerable in principle?
- answer_completeness: Does the answer fully address the question?
  Factoid: is it precise? Multi-hop: does it combine both pieces?
  Abstractive: does it paraphrase correctly? Unanswerable: is the prescribed answer used?
- overall: Overall quality as a training example (correctness, usefulness, naturalness).

Return JSON:
{{
  "groundedness": <0-10>,
  "question_clarity": <0-10>,
  "answer_completeness": <0-10>,
  "overall": <0-10>,
  "reasoning": "<1-2 sentence justification>"
}}"""
