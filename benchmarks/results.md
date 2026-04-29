# Benchmark Results

All experiments fine-tune `unsloth/Llama-3.2-3B-Instruct` with LoRA (r=16) for 3 epochs.
Tessera generation uses `gpt-4o-mini` with critique threshold 6.0.

---

## Banking-77 Classification (5-class subset)

| Condition | Train Size | F1 (macro) | Accuracy | % of Real |
|-----------|-----------|-----------|----------|-----------|
| Tessera synthetic | 1000 | [TBD] | [TBD] | [TBD]% |
| Real labelled data | 1000 | [TBD] | [TBD] | 100% |
| Random baseline | 0 | [TBD] | 20.0% | — |

**Labels:** card_lost, transfer_failed, balance_inquiry, account_locked, foreign_transaction_declined

---

## DocRED Relation Extraction

| Condition | Train Size | Per-field F1 | JSON Validity | % of Real |
|-----------|-----------|-------------|--------------|-----------|
| Tessera synthetic | 500 | [TBD] | [TBD]% | [TBD]% |
| Real labelled data | 500 | [TBD] | 100% | 100% |

**Schema fields:** subject, relation, object, sentence

---

## Python Instruction-Following

| Condition | Train Size | LLM Judge (0-10) | % of Real |
|-----------|-----------|-----------------|-----------|
| Tessera synthetic | 500 | [TBD] | [TBD]% |
| Real (Stack Overflow curated) | 500 | [TBD] | 100% |

**Instruction types:** write function, debug code, explain concept, refactor code

---

## Pipeline Statistics

| Task | Generated | After Critique | After Dedup | Pass Rate | Est. Cost |
|------|-----------|---------------|-------------|----------|-----------|
| Banking-77 (1000 target) | [TBD] | [TBD] | [TBD] | [TBD]% | $[TBD] |
| DocRED (500 target) | [TBD] | [TBD] | [TBD] | [TBD]% | $[TBD] |
| Python Instr. (500 target) | [TBD] | [TBD] | [TBD] | [TBD]% | $[TBD] |

---

## Ablation: Critique Threshold

*(Banking-77, 1000 examples)*

| Threshold | Pass Rate | Final Count | F1 (macro) |
|-----------|----------|-------------|-----------|
| 5.0 | [TBD]% | [TBD] | [TBD] |
| 6.0 | [TBD]% | [TBD] | [TBD] |
| 7.0 | [TBD]% | [TBD] | [TBD] |

---

## Ablation: Dedup Threshold

*(Banking-77, 1000 examples)*

| Threshold | Removed | Final Count | F1 (macro) |
|-----------|---------|-------------|-----------|
| 0.85 | [TBD] | [TBD] | [TBD] |
| 0.90 | [TBD] | [TBD] | [TBD] |
| 0.95 | [TBD] | [TBD] | [TBD] |
| No dedup | 0 | [TBD] | [TBD] |

---

*Results will be filled in after Colab GPU runs. See LOG.md Week 2.*
