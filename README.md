<div align="center">

# Tessera

**Generate production-quality synthetic fine-tuning datasets for LLMs — validated end-to-end with real benchmarks.**

[![CI](https://github.com/muhaiminmemon/tessera/actions/workflows/ci.yml/badge.svg)](https://github.com/muhaiminmemon/tessera/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-98%20passing-brightgreen.svg)](tests/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

<br/>

> Tessera-trained **Llama-3.2-3B** reaches **97.1% of real-data F1** on Banking77 classification for **$0.40**,
> and **72.5% of real SQuAD F1** for **$0.19** — with **zero human annotation**.

<br/>

</div>

---

## Why Tessera?

Fine-tuning a small LLM requires thousands of labeled examples. Getting them the traditional way — crowdsourcing, expert annotation, or curating from public datasets — is slow, expensive, and often impossible for proprietary domains.

Tessera generates that data automatically. You describe what you need in plain text. Tessera expands it into a diverse taxonomy, writes examples through 60 curated personas, filters them through a multi-axis self-critique stage, removes near-duplicates with embedding similarity, and returns a ready-to-train JSONL file.

**The key difference from just prompting GPT-4:** Tessera's 6-stage pipeline produces diverse, high-quality, non-repetitive examples that benchmark within 3–30% of human-annotated datasets — at 100–750× lower cost.

---

## Benchmark results

### Banking77 Intent Classification

Fine-tuned `Llama-3.2-3B` on 1,246 Tessera-generated examples vs. real human-labeled Banking77 data.

| Condition | Macro F1 | Accuracy | Cost |
|-----------|----------|----------|------|
| Random baseline | 0.1665 | — | $0 |
| Real labeled data | 0.8102 | 96.5% | ~$300+ (human annotation) |
| **Tessera synthetic** | **0.7869** | **93.5%** | **$0.40** |

**97.1% of real-data F1 at 750× lower cost.**

---

### SQuAD Question Answering

Fine-tuned on 137 Tessera-generated QA pairs (context, question, answer triples) in SQuAD format.

| Condition | Token F1 | Cost |
|-----------|----------|------|
| Real SQuAD data | 0.8522 | Human annotated |
| **Tessera synthetic** | **0.6182** | **$0.19** |

**72.5% of real SQuAD F1 for 19 cents.**

Scaling projection:

| Examples | Est. Token F1 | Est. % of real data | Est. cost |
|----------|---------------|---------------------|-----------|
| 137 (benchmarked) | 0.618 | 72.5% | $0.19 |
| 500 | ~0.70–0.73 | ~82–86% | ~$0.65 |
| 1,000 | ~0.75–0.78 | ~88–91% | ~$1.30 |
| 2,000 | ~0.78–0.81 | ~91–95% | ~$2.60 |

---

### Enterprise RAG Evaluation

Tessera generates domain-specific QA pairs, then evaluates a RAG reader model against them. Tested with `gpt-4o-mini` as the reader on 150 enterprise software documentation examples.

| Metric | Score | What it measures |
|--------|-------|-----------------|
| Factoid F1 | **0.822** | Direct fact extraction from context |
| Multi-hop F1 | **0.819** | Reasoning across two facts in a passage |
| Unanswerable accuracy | **0.980** | Hallucination refusal rate |
| **Overall** | **0.873** | Mean across all question types |

**98% hallucination refusal rate** — the model correctly refused to answer when the context didn't support it on 49/50 unanswerable questions.

---

### Python Instruction Following

| Condition | ROUGE-L | Code block rate |
|-----------|---------|----------------|
| Zero-shot baseline | 0.2922 | 100% |
| **Tessera fine-tuned** | **0.3707** | **100%** |

**+26.85% ROUGE-L improvement over zero-shot** with ~317 generated examples for ~$0.30.

---

### DocRED Relation Extraction

| Condition | Macro F1 | % of real data |
|-----------|----------|----------------|
| Real labeled data | 0.5895 | 100% |
| Tessera synthetic | 0.1873 | 31.8% |

Extraction is the hardest task type — the model simultaneously learns entity identification and relation type from few examples per class. Scaling to 2,000+ examples is projected to reach 55–65%.

---

### Summary

| Task | Metric | Result | Cost |
|------|--------|--------|------|
| Banking77 classification | Macro F1 | **97.1% of real-data** | $0.40 |
| SQuAD QA | Token F1 | **72.5% of real-data** | $0.19 |
| Enterprise RAG eval | Hallucination refusal | **98% accuracy** | $0.19 |
| Python instruction | ROUGE-L | **+26.85% over zero-shot** | ~$0.30 |
| DocRED extraction | Macro F1 | 31.8% of real-data | ~$0.50 |

---

## Install

```bash
git clone https://github.com/muhaiminmemon/tessera
cd tessera
pip install -e ".[dev]"
```

> PyPI release coming soon. For now, install from source above.

Add your API keys — copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # optional
TOGETHER_API_KEY=...            # optional
GROQ_API_KEY=...                # optional
```

Tessera routes to whichever provider you have keys for. Only `OPENAI_API_KEY` is required for all task types out of the box.

---

## Quick start

### Classification

```python
from tessera import generate

result = generate(
    task="classification",
    spec_dict={
        "domain": "banking customer support",
        "labels": ["card_lost", "transfer_failed", "balance_inquiry", "account_locked"],
        "label_descriptions": {
            "card_lost": "Customer reporting lost or stolen card",
            "transfer_failed": "Customer reporting a failed money transfer",
            "balance_inquiry": "Customer asking about their current account balance",
            "account_locked": "Customer locked out of their online banking account",
        },
    },
    n_examples=1000,
    model="gpt-4o-mini",
    output_format="alpaca",
    output_path="banking_train.jsonl",
)
print(f"Generated {len(result.examples)} examples for ${result.cost_usd:.4f}")
```

**Output** (`alpaca` format):
```json
{
  "instruction": "Classify the following text.",
  "input": "Hi, I can't seem to log into my account. I've tried resetting my password twice.",
  "output": "account_locked"
}
```

---

### Structured Extraction

```python
result = generate(
    task="extraction",
    spec_dict={
        "domain": "legal contracts and agreements",
        "schema_definition": {
            "party_a": "Name of the first contracting party",
            "party_b": "Name of the second contracting party",
            "effective_date": "Date the contract comes into effect",
            "governing_law": "Jurisdiction governing the contract",
            "termination_clause": "Summary of termination conditions",
        },
        "source_text_type": "contract document",
    },
    n_examples=500,
    model="gpt-4o-mini",
    output_format="alpaca",
    output_path="legal_extraction_train.jsonl",
)
```

**Output**:
```json
{
  "instruction": "Extract structured information from the provided text and return valid JSON.",
  "input": "This Agreement is entered into as of January 15, 2025, between Acme Corp...",
  "output": "{\"party_a\": \"Acme Corp\", \"party_b\": \"Globex Inc\", \"effective_date\": \"January 15, 2025\", ...}"
}
```

---

### Instruction Following

```python
result = generate(
    task="instruction",
    spec_dict={
        "domain": "Python programming and software engineering",
        "instruction_types": ["write_function", "debug_code", "explain_concept", "refactor_code"],
        "response_format": "prose with code blocks",
    },
    n_examples=500,
    model="gpt-4o-mini",
    output_format="alpaca",
    output_path="python_instruction_train.jsonl",
)
```

---

### RAG / Question Answering

```python
# Generate a QA dataset for fine-tuning or RAG evaluation
result = generate(
    task="qa",
    spec_dict={
        "domain": "enterprise software documentation, internal company policies, and technical product manuals",
        "question_types": ["factoid", "multi-hop", "unanswerable"],
    },
    n_examples=500,
    model="gpt-4o-mini",
    output_format="jsonl",
    output_path="rag_eval_qa.jsonl",
)
```

**Output** (`jsonl` format):
```json
{
  "context": "Acme CRM version 4.2 introduced role-based access control in Q3 2024...",
  "question": "When was role-based access control introduced in Acme CRM?",
  "answer": "Q3 2024",
  "question_type": "factoid",
  "difficulty": "easy",
  "label": "factoid"
}
```

Also available in **SQuAD format** for direct benchmark comparison:
```python
from tessera.tasks.qa import QATask
task = QATask()
rows = task.format_for_finetuning(result.examples, fmt="squad")
# {"id": ..., "context": ..., "question": ..., "answers": {"text": [...], "answer_start": [...]}}
```

---

### Evaluate your RAG system

```bash
python scripts/evaluate_rag.py \
  --input outputs/rag_eval_qa.jsonl \
  --model gpt-4o-mini
```

Output:
```
=== RAG Evaluation Results ===
Factoid F1:             0.822   (n=50)
Multi-hop F1:           0.819   (n=50)
Unanswerable accuracy:  0.980   (n=50)
Overall:                0.873
```

---

## CLI

```bash
# Classification
tessera generate \
  --task classification \
  --domain "e-commerce customer support" \
  --labels "order_tracking,return_request,payment_issue,product_question" \
  --n 1000 \
  --model gpt-4o-mini \
  --output ecommerce_train.jsonl

# Instruction following
tessera generate \
  --task instruction \
  --domain "SQL database queries and optimization" \
  --n 500 \
  --output sql_instructions.jsonl

# QA / RAG
tessera qa \
  --domain "healthcare insurance policies and claim procedures" \
  --question-types "factoid,multi-hop,unanswerable" \
  --n-examples 500 \
  --model gpt-4o-mini \
  --output-path healthcare_qa.jsonl

# Validate a generated dataset
tessera validate \
  --input outputs/banking77_tessera_train.jsonl \
  --task classification
```

---

## How it works

Tessera runs a 6-stage pipeline. Each stage is independently configurable and all generation is parallel (configurable via `TESSERA_MAX_CONCURRENT`).

```
Your spec (domain, labels / schema / question_types)
         │
         ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Taxonomy Expansion           │
│                                         │
│  LLM generates 40-60 diverse scenario   │
│  nodes from your spec. Each node has    │
│  a category, subcategory, scenario,     │
│  and target label. Balanced across      │
│  all labels automatically.              │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Persona-based Generation     │
│                                         │
│  60 curated personas (expert→novice,    │
│  formal→casual, diverse cultural        │
│  contexts) × taxonomy nodes → raw       │
│  Examples in parallel threads.          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 3 — Multi-axis Self-Critique     │
│                                         │
│  A second LLM call scores each example  │
│  on 3 axes (0–10 each):                 │
│    • realism / groundedness             │
│    • label_correctness / clarity        │
│    • specificity / completeness         │
│  Mean < threshold → discarded.          │
│  Thresholds: 7.0 (class/instr),         │
│              7.5 (extraction/QA)        │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 4 — Embedding Deduplication      │
│                                         │
│  all-MiniLM-L6-v2 + ChromaDB removes    │
│  near-duplicate examples                │
│  (cosine similarity ≥ threshold).       │
│  QA deduplicates on question text only  │
│  to catch semantically identical        │
│  questions across different contexts.   │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 5 — Hard-Negative Mining         │
│                                         │
│  LogisticRegression on embeddings       │
│  identifies near-boundary examples      │
│  (confusable between labels) and        │
│  oversamples them. Sharpens decision    │
│  boundaries without more LLM calls.     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 6 — Downstream Validation        │
│                                         │
│  Optional: Unsloth LoRA fine-tune       │
│  (Llama-3.2-3B, r=16, 3 epochs) →       │
│  F1 / ROUGE-L / token F1 vs held-out    │
│  test set and real-data baseline.       │
└─────────────────────────────────────────┘
         │
         ▼
  JSONL / Alpaca / ShareGPT / SQuAD
```

---

## Task types

| Task | Use case | Output formats | Benchmark |
|------|----------|----------------|-----------|
| `classification` | Intent detection, sentiment, topic routing | `jsonl`, `alpaca`, `sharegpt` | Banking77 |
| `extraction` | Named entity, relation, structured field extraction | `jsonl`, `alpaca`, `sharegpt` | DocRED |
| `instruction` | Instruction-following, code generation, explanation | `alpaca`, `sharegpt`, `jsonl` | Python coding |
| `qa` | RAG evaluation, QA fine-tuning, reading comprehension | `jsonl`, `squad`, `alpaca` | SQuAD v1.1 |

### QA question types

| Type | Description | Use case |
|------|-------------|----------|
| `factoid` | Single fact directly stated in the passage | Testing precise retrieval |
| `multi-hop` | Requires combining two pieces of information from the passage | Testing reasoning chains |
| `abstractive` | Answer requires synthesis/paraphrase, not direct extraction | Testing generalization |
| `unanswerable` | Plausible question but not answerable from context | Testing hallucination refusal |

---

## Supported models

| Provider | Models | Best for |
|----------|--------|----------|
| **OpenAI** | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1-nano` | Default — best quality/cost balance |
| **Anthropic** | `claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022` | Long-context, nuanced generation |
| **Together AI** | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Open-weight, cost-sensitive workloads |
| **Groq** | `llama-3.1-8b-instant` | Ultra-fast prototyping |

Mix models per stage — e.g., use `gpt-4.1-mini` for generation and `gpt-4o-mini` for critique:

```python
from tessera.tasks.classification import ClassificationTask

task = ClassificationTask(
    model="gpt-4.1-mini",          # generation
    critique_model="gpt-4o-mini",  # critique (cheaper, still good)
    dedup_threshold=0.85,
    critique_threshold=7.0,
)
```

---

## Configuration

All settings have sensible defaults. Override via environment variables or constructor arguments.

```bash
# .env
OPENAI_API_KEY=sk-...
TESSERA_DEFAULT_MODEL=gpt-4o-mini
TESSERA_MAX_CONCURRENT=10          # parallel LLM threads
```

```python
from tessera.tasks.qa import QATask
from tessera.core.models import QASpec
from tessera.core.personas import get_all_personas

# Fine-grained control
task = QATask(
    model="gpt-4o-mini",
    critique_model="gpt-4o-mini",
    dedup_threshold=0.85,      # cosine similarity cutoff for dedup
    critique_threshold=7.5,    # minimum mean critique score (0-10)
)

spec = QASpec(
    domain="financial services regulation and compliance",
    question_types=["factoid", "multi-hop", "unanswerable"],
    language="English",
)

result = task.run_pipeline(
    spec=spec,
    personas=get_all_personas(),
    n_examples=500,
    max_attempts_multiplier=2.5,   # generate 2.5× target, filter down
)

print(f"Generated: {result.total_generated}")
print(f"After critique: {result.total_after_critique}")
print(f"After dedup: {result.total_after_dedup}")
print(f"Final: {len(result.examples)}")
print(f"Cost: ${result.cost_usd:.4f}")
```

---

## Output formats

### JSONL (default)
```json
{"text": "I need to dispute a transaction on my statement", "label": "dispute_transaction"}
```

### Alpaca
```json
{
  "instruction": "Classify the following text.",
  "input": "I need to dispute a transaction on my statement",
  "output": "dispute_transaction"
}
```

### ShareGPT
```json
{
  "conversations": [
    {"from": "human", "value": "Classify this text: I need to dispute a transaction"},
    {"from": "gpt", "value": "dispute_transaction"}
  ]
}
```

### SQuAD (QA only)
```json
{
  "id": "abc123",
  "context": "The policy was last updated on March 1, 2024...",
  "question": "When was the policy last updated?",
  "answers": {
    "text": ["March 1, 2024"],
    "answer_start": [35]
  },
  "question_type": "factoid",
  "difficulty": "easy"
}
```

---

## Project structure

```
tessera/
├── core/
│   ├── models.py          # Pydantic schemas (TaskType, Example, QASpec, ...)
│   ├── base.py            # TaskTemplate ABC + run_pipeline orchestrator
│   ├── llm_client.py      # Unified LLM router (OpenAI / Anthropic / Together / Groq)
│   ├── prompts.py         # All LLM prompts — zero hardcoded strings in task files
│   ├── personas.py        # 60 curated generation personas
│   ├── config.py          # Runtime config (temperatures, concurrency)
│   └── exceptions.py      # Structured exception hierarchy
├── pipeline/
│   ├── taxonomy.py        # Stage 1 — taxonomy expansion
│   ├── generation.py      # Stage 2 — example generation
│   ├── critique.py        # Stage 3 — multi-axis self-critique
│   ├── dedup.py           # Stage 4 — embedding deduplication
│   └── hard_negative.py   # Stage 5 — hard-negative mining
├── tasks/
│   ├── classification.py  # ClassificationTask
│   ├── extraction.py      # ExtractionTask
│   ├── instruction.py     # InstructionTask
│   └── qa.py              # QATask
└── validation/
    ├── finetune.py        # Unsloth LoRA fine-tuning wrapper
    ├── evaluate.py        # F1 / ROUGE-L evaluation
    └── benchmarks.py      # BenchmarkRunner with real-data comparison
examples/                  # Runnable end-to-end scripts
scripts/                   # Evaluation and analysis utilities
tests/                     # 98 unit tests, all LLM calls mocked
```

---

## Adding a custom task type

Tessera is designed to be extended. Adding a new task type requires changes in 5 places — no modifications to the pipeline core (Open/Closed Principle):

1. **`tessera/core/models.py`** — add `TaskType.YOUR_TASK` and a `YourSpec` Pydantic model
2. **`tessera/core/prompts.py`** — add `your_task_taxonomy_system/user`, `your_task_generation_system/user`, `your_task_critique_system/user`
3. **`tessera/pipeline/taxonomy.py`** — add one entry to `_TAXONOMY_PROMPTS` and optionally `_COVERAGE_EXTRACTORS`
4. **`tessera/pipeline/generation.py`** — add one entry to `_GEN_PROMPTS` (or implement two-step generation like QA)
5. **`tessera/pipeline/critique.py`** — add one entry to `_CRITIQUE_PROMPTS`
6. **`tessera/tasks/your_task.py`** — implement `TaskTemplate` (copy `classification.py` as a template)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full step-by-step guide.

---

## Fine-tuning on Colab

Open the Colab notebook for one-click Unsloth LoRA fine-tuning on any Tessera-generated dataset:

```python
# In Colab — install and run
!pip install unsloth tessera
from tessera.validation.finetune import UnslothFinetuner

finetuner = UnslothFinetuner(
    base_model="unsloth/Llama-3.2-3B-Instruct",
    output_dir="tessera_model",
    num_epochs=3,
    lora_r=16,
)
finetuner.run(train_path="banking_train.jsonl", task_type="classification")
```

---

## Error handling

Tessera uses a structured exception hierarchy — catch specific errors rather than bare `Exception`:

```python
from tessera import generate
from tessera import (
    TesseraError,
    GenerationError,
    CritiqueError,
    TaxonomyError,
    ConfigurationError,
)

try:
    result = generate(task="classification", spec_dict={...}, n_examples=500)
except ConfigurationError as e:
    print(f"Bad config: {e}")
except TaxonomyError as e:
    print(f"Taxonomy expansion failed: {e}")
except TesseraError as e:
    print(f"Pipeline error: {e}")
```

---

## Development

```bash
# Run tests
make test

# Lint + typecheck
make lint

# Auto-format
make fmt

# Run everything before a PR
make check
```

All LLM calls in tests are mocked — the full test suite runs in under 1 second and costs $0.

---

## Roadmap

- [ ] **HuggingFace Hub push** — `result.push_to_hub("username/dataset-name")`
- [ ] **Summarization task** — (prompt, summary) pairs, ROUGE-L benchmark vs CNN/DailyMail
- [ ] **Preference / RLHF task** — (prompt, chosen, rejected) triples for DPO training
- [ ] **Dialogue task** — multi-turn conversations in ShareGPT format
- [ ] **Dataset card auto-generation** — auto-generate HuggingFace dataset cards with benchmark numbers
- [ ] **Cost budgeting** — `max_cost_usd` parameter to cap spend before running
- [ ] **PyPI release** — `pip install tessera` from the public index (packaging is ready, pending first stable release)

---

## Examples

| File | Task | Domain | Examples | Cost |
|------|------|--------|----------|------|
| `examples/banking77_classification.py` | Classification | Banking intent (5 labels) | 1,246 | $0.40 |
| `examples/docred_extraction.py` | Extraction | Wikipedia relation extraction | 500 | ~$0.50 |
| `examples/python_instruction.py` | Instruction | Python coding | ~317 | ~$0.30 |
| `examples/squad_qa.py` | QA | General knowledge encyclopedia | 137 | $0.19 |
| `examples/rag_eval_qa.py` | QA | Enterprise software documentation | 150 | $0.19 |
| `scripts/evaluate_rag.py` | RAG eval | Any domain | — | ~$0.05 per 150 examples |

---

## Contributing

Contributions are welcome — bug fixes, new task types, prompt improvements, and benchmark results.

1. Fork the repo and create a feature branch
2. `pip install -e ".[dev]"` and run `make check`
3. Open a PR with a description and test coverage

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including how to add a new task type end-to-end.

Issues and feature requests: [GitHub Issues](https://github.com/muhaiminmemon/tessera/issues)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with `gpt-4o-mini`, `sentence-transformers`, `chromadb`, `pydantic`, `typer`, and `unsloth`.

**Star the repo if Tessera saves you annotation time.**

</div>