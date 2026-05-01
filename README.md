# Tessera

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)

**Tessera generates high-quality synthetic training datasets for fine-tuning small LLMs — validated end-to-end with real benchmark experiments.**

> Tessera-trained Llama-3.2-3B reaches **97.1% of real-data F1** on Banking77 for $0.40, and **72.5% of real SQuAD F1** for $0.19 — with zero human annotation.

---

## Install

```bash
pip install tessera
```

Or from source:

```bash
git clone https://github.com/muhaiminmemon/tessera
cd tessera
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and add your API keys.

---

## Quick start

```python
from tessera import generate

# Classification
result = generate(
    task="classification",
    spec_dict={
        "domain": "banking customer support",
        "labels": ["card_lost", "transfer_failed", "balance_inquiry"],
        "label_descriptions": {
            "card_lost": "Customer reporting lost or stolen card",
            "transfer_failed": "Customer reporting a failed money transfer",
            "balance_inquiry": "Customer asking about their balance",
        },
    },
    n_examples=1000,
    model="gpt-4o-mini",
    output_format="alpaca",
    output_path="banking_train.jsonl",
)
print(f"Generated {len(result.examples)} examples for ${result.cost_usd:.4f}")
```

```python
# RAG / QA
result = generate(
    task="qa",
    spec_dict={
        "domain": "enterprise software documentation and internal company policies",
        "question_types": ["factoid", "multi-hop", "unanswerable"],
    },
    n_examples=500,
    model="gpt-4o-mini",
    output_format="jsonl",
    output_path="rag_eval_qa.jsonl",
)
print(f"Generated {len(result.examples)} QA pairs for ${result.cost_usd:.4f}")
```

CLI:

```bash
# Classification
tessera generate \
  --task classification \
  --domain "banking customer support" \
  --labels "card_lost,transfer_failed,balance_inquiry" \
  --n 1000 \
  --output banking_train.jsonl

# RAG / QA
tessera qa \
  --domain "enterprise software documentation" \
  --n-examples 500 \
  --output-path rag_eval_qa.jsonl \
  --fmt jsonl
```

---

## Architecture

```
Task Spec (domain, labels / schema / instruction_types / question_types)
         │
         ▼
┌─────────────────────┐
│  1. Taxonomy        │  LLM expands spec into N diverse scenario nodes
│     Expansion       │  (≥3 nodes per label, varied styles & contexts)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Persona-based   │  60 curated personas × taxonomy nodes
│     Generation      │  → raw Examples (text+label / source+fields /
│                     │    instruction+response / context+question+answer)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Multi-axis      │  LLM scores each example on realism,
│     Self-Critique   │  label_correctness, specificity (0-10 each)
└────────┬────────────┘
         │  filter mean ≥ threshold (7.0–7.5 depending on task)
         ▼
┌─────────────────────┐
│  4. Embedding       │  sentence-transformers + ChromaDB
│     Dedup           │  drops cosine similarity ≥ 0.90
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Hard-Negative   │  LogisticRegression on embeddings,
│     Mining          │  oversample near-boundary examples
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Downstream      │  Unsloth LoRA fine-tune → F1/accuracy
│     Validation      │  vs real data and random baseline
└─────────────────────┘
```

---

## Task types

| Task | Benchmark | Primary Metric | Output Format | Status |
|------|-----------|---------------|---------------|--------|
| Classification | Banking-77 | F1 (macro) | jsonl / alpaca / sharegpt | ✅ Alpha |
| Structured Extraction | DocRED | Per-field F1 | jsonl / alpaca | ✅ Alpha |
| Instruction-following | Custom Python | ROUGE-L | alpaca / sharegpt | ✅ Alpha |
| RAG / QA | SQuAD v1.1 | Token F1 | jsonl / squad / alpaca | ✅ Alpha |

### QA question types

| Type | Description | Example |
|------|-------------|---------|
| `factoid` | Single fact directly stated in the passage | "What year was the policy last updated?" |
| `multi-hop` | Requires combining two pieces of information | "Who approved the merger and in what year?" |
| `abstractive` | Answer requires synthesis, not direct extraction | "Why did the company restructure its divisions?" |
| `unanswerable` | Plausible question but answer is absent from context | Answer: *"This cannot be determined from the provided context."* |

---

## Pipeline stages

1. **Taxonomy Expansion** — LLM generates a tree of diverse, concrete scenarios per label
2. **Persona-based Generation** — 60 personas (expert → novice, formal → very casual) write examples for each scenario
3. **Multi-axis Self-Critique** — second LLM pass scores realism / label-correctness / specificity; low-scoring examples discarded (threshold 7.0–7.5)
4. **Embedding Dedup** — `all-MiniLM-L6-v2` + ChromaDB removes near-duplicates (cosine ≥ 0.90)
5. **Hard-Negative Mining** — logistic regression on embeddings identifies near-boundary examples and oversamples them
6. **Downstream Validation** — Unsloth LoRA fine-tunes Llama-3.2-3B and measures F1 against held-out test set

---

## Benchmark results

### Task 1 — Banking77 Intent Classification

| Condition | Macro F1 | Accuracy | Training examples | Cost |
|-----------|----------|----------|------------------|------|
| Random baseline | 0.1665 | — | — | — |
| Real labeled data | 0.8102 | 96.5% | 1,246 | ~$300+ (human annotation) |
| **Tessera synthetic** | **0.7869** | **93.5%** | **1,246** | **$0.40** |

**Tessera = 97.1% of real-data F1 at 155× lower cost**

---

### Task 2 — SQuAD Question Answering (RAG evaluation)

| Condition | Token F1 | Training examples | Cost |
|-----------|----------|------------------|------|
| Real SQuAD data | 0.8522 | 137 | Human annotated |
| **Tessera synthetic** | **0.6182** | **137** | **$0.19** |

**Tessera = 72.5% of real SQuAD F1 for $0.19**

#### Scaling projection (estimated)

| Training examples | Est. Token F1 | Est. % of real data | Est. cost |
|------------------|---------------|---------------------|-----------|
| 137 (benchmarked) | 0.618 | 72.5% | $0.19 |
| 500 | ~0.70–0.73 | ~82–86% | ~$0.65 |
| 1,000 | ~0.75–0.78 | ~88–91% | ~$1.30 |
| 2,000 | ~0.78–0.81 | ~91–95% | ~$2.60 |

---

### Task 3 — Python Instruction Following

| Condition | ROUGE-L | Code block rate |
|-----------|---------|----------------|
| Zero-shot baseline | 0.2922 | 100% |
| **Tessera fine-tuned** | **0.3707** | **100%** |

**+26.85% ROUGE-L improvement over zero-shot**

---

### Task 4 — DocRED Relation Extraction

| Condition | Macro F1 | % of real data |
|-----------|----------|----------------|
| Real labeled data | 0.5895 | 100% |
| Tessera synthetic | 0.1873 | 31.8% |

**Honest framing:** extraction is structurally harder than classification — the model must simultaneously learn entity pair identification and relation type from ~62 examples per relation. This is an active area of improvement. Scaling to 2,000+ balanced examples is expected to push this to 55–65%.

---

### Summary

| Task | Tessera % of real data | Training cost |
|------|----------------------|---------------|
| Banking77 classification | **97.1%** | $0.40 |
| SQuAD QA | **72.5%** | $0.19 |
| Python instruction | **+26.85% ROUGE-L** | ~$0.30 |
| DocRED extraction | 31.8% | ~$0.50 |

---

## Supported models

| Provider | Example models |
|----------|---------------|
| OpenAI | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1-nano` |
| Anthropic | `claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022` |
| Together AI | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Groq | `llama-3.1-8b-instant` |

---

## Examples

| File | Task | Domain | Examples | Cost |
|------|------|--------|----------|------|
| `examples/banking77_classification.py` | Classification | Banking intent | 1,246 | $0.40 |
| `examples/docred_extraction.py` | Extraction | Wikipedia relations | 500 | ~$0.50 |
| `examples/python_instruction.py` | Instruction | Python coding | ~317 | ~$0.30 |
| `examples/rag_eval_qa.py` | QA | Enterprise docs | 150 | $0.19 |
| `examples/squad_qa.py` | QA | General knowledge | 137 | $0.19 |

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `pip install -e ".[dev]"` and `pytest`
3. Follow the `ruff` style guide (`ruff check .`)
4. Open a PR — please include a short description and test coverage

Issues and feature requests: [GitHub Issues](https://github.com/muhaiminmemon/tessera/issues)
