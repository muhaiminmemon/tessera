# Tessera — Vision & Roadmap

## What Tessera Is Solving

Acquiring labelled training data for fine-tuning small LLMs is the bottleneck in most enterprise ML projects.  Human annotation is slow, expensive, and hard to scale.  Tessera's thesis: a well-designed synthetic data pipeline — diverse taxonomy, persona-grounded generation, multi-axis critique, and hard-negative mining — can close most of the gap to real data at a fraction of the cost.

---

## v1 (current)

Three task types:
- **Classification** — intent detection, routing, sentiment
- **Structured Extraction** — relation triples, form parsing, slot filling
- **Instruction-following** — domain-specific Q&A and coding assistants

Single-model orchestration (one LLM per stage), CLI + Python API, Unsloth LoRA validation.

---

## v2 Task Types

### Dialogue / Multi-turn
Generate realistic multi-turn conversations for chatbot fine-tuning.  Each dialogue is seeded from a persona + scenario + resolution arc.  Critique evaluates turn coherence and goal completion.

### Summarisation
Generate (document, summary) pairs at varying compression ratios and formality levels.  Hard negatives = near-miss summaries with subtle factual errors, evaluated by ROUGE + LLM judge.

### Reasoning Chains (Chain-of-Thought)
Generate step-by-step reasoning traces for math, logic, and commonsense problems.  Critique scores correctness of intermediate steps, not just the final answer.

---

## Active Learning Loop (v2)

```
Train small model on Tessera batch
        │
        ▼
Evaluate on unlabelled pool
        │
        ▼
Identify high-uncertainty examples (entropy sampling)
        │
        ▼
Generate targeted synthetic examples in uncertain regions
        │
        ▼
Repeat
```

This closes the feedback loop: Tessera gets smarter about where to generate as the downstream model improves.

---

## Multi-model Orchestration

**Why use multiple models?**

- **Generator diversity**: GPT-4o-mini writes differently than Llama-70B; mixing providers increases surface coverage.
- **Cross-model critique**: Having Anthropic Claude score OpenAI-generated examples reduces style-specific blind spots.
- **Cost control**: Route cheap/easy examples to smaller models, send edge cases to frontier models.

Planned v2 API:
```python
result = generate(
    task="classification",
    spec_dict=spec,
    n_examples=1000,
    generator_models=["gpt-4o-mini", "meta-llama/Llama-3.3-70B"],
    critique_model="claude-3-5-sonnet-20241022",
)
```

---

## HuggingFace Dataset Cards

Generated datasets will be publishable directly to the HuggingFace Hub with auto-generated dataset cards including:
- Generation spec (domain, labels, model, cost)
- Pipeline statistics (critique pass rate, dedup rate)
- Benchmark results comparing to real data baselines
- Example samples per label/category

---

## Long-term: Web UI

A no-code interface where non-engineers can:
1. Describe their task in natural language
2. Select from suggested label sets or schemas
3. Preview samples before committing to full generation
4. Monitor pipeline progress in real time
5. Download the final dataset in any format

Target users: ML engineers at startups who need labelled data fast, researchers running ablations, and product teams building domain-specific assistants.

---

## Principles

- **Every claim is benchmarked.** No marketing without numbers.
- **Composable stages.** Each pipeline step is a separate class you can swap out.
- **API-first.** The Python API is the source of truth; CLI and UI are thin wrappers.
- **Honest about limits.** Tessera won't replace real data entirely — it gets you 70-90% there faster.
