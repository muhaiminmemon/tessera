# Tessera

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)

**Tessera generates high-quality synthetic training datasets for fine-tuning small LLMs — validated end-to-end with real benchmark experiments.**

> Tessera-trained Llama-3.2-3B reaches **[X]%** of real-data baseline F1 at 1/30th the data acquisition cost, validated on Banking77, DocRED, and custom Python tasks.

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

print(f"Generated {len(result.examples)} examples")
print(f"Cost: ${result.cost_usd:.4f}")
```

CLI:

```bash
tessera generate \
  --task classification \
  --domain "banking customer support" \
  --labels "card_lost,transfer_failed,balance_inquiry" \
  --n 1000 \
  --output banking_train.jsonl \
  --format alpaca
```

---

## Architecture

```
Task Spec (domain, labels/schema/instruction_types)
         │
         ▼
┌─────────────────────┐
│  1. Taxonomy        │  LLM expands spec into N diverse scenario nodes
│     Expansion       │  (≥3 nodes per label, varied styles & contexts)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Persona-based   │  50 curated personas × taxonomy nodes
│     Generation      │  → raw Examples (text+label / source+fields / instr+response)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Multi-axis      │  LLM scores each example on realism,
│     Self-Critique   │  label_correctness, specificity (0-10 each)
└────────┬────────────┘
         │  filter mean ≥ threshold (default 6.0)
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

| Task | Benchmark | Primary Metric | Status |
|------|-----------|---------------|--------|
| Classification | Banking-77 | F1 (macro) | Alpha |
| Structured Extraction | DocRED | Per-field F1 | Alpha |
| Instruction-following | Custom Python | LLM judge (0-10) | Alpha |

---

## Pipeline stages

1. **Taxonomy Expansion** — LLM generates a tree of diverse, concrete scenarios per label
2. **Persona-based Generation** — 50 personas (expert → novice, formal → very casual) write examples for each scenario
3. **Multi-axis Self-Critique** — second LLM pass scores realism / label-correctness / specificity; low-scoring examples discarded
4. **Embedding Dedup** — `all-MiniLM-L6-v2` + ChromaDB removes near-duplicates (cosine ≥ 0.90)
5. **Hard-Negative Mining** — logistic regression on embeddings identifies near-boundary examples and oversamples them
6. **Downstream Validation** — Unsloth LoRA fine-tunes Llama-3.2-3B and measures F1 against held-out test set

---

## Benchmark results

| Task | Tessera F1 | Real Data F1 | Random Baseline | % of Real |
|------|-----------|-------------|----------------|-----------|
| Banking-77 (5-class) | [TBD] | [TBD] | [TBD] | [TBD]% |
| DocRED Extraction | [TBD] | [TBD] | N/A | [TBD]% |
| Python Instructions | [TBD] | [TBD] | N/A | [TBD]% |

See [benchmarks/results.md](benchmarks/results.md) for full results.

---

## Supported models

| Provider | Example models |
|----------|---------------|
| OpenAI | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022` |
| Together AI | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Groq | `llama-3.1-8b-instant` |

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `pip install -e ".[dev]"` and `pytest`
3. Follow the `ruff` style guide (`ruff check .`)
4. Open a PR — please include a short description and test coverage

Issues and feature requests: [GitHub Issues](https://github.com/muhaiminmemon/tessera/issues)
