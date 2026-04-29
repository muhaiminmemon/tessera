# Tessera Build Log

## Week 1 — Foundation (2026-04-28)

**Goals:**
- [x] Project scaffold: `pyproject.toml`, `.gitignore`, `.env.example`
- [x] Core models: `TaskType`, `ClassificationSpec`, `ExtractionSpec`, `InstructionSpec`, `Taxonomy`, `Persona`, `Example`, `CritiqueScores`, `GenerationResult`, `ValidationMetrics`
- [x] 50 personas covering the full formality × expertise matrix
- [x] Unified LLM client routing to OpenAI / Anthropic / Together / Groq
- [x] Prompt library for all three task types × all three pipeline stages
- [x] Abstract `TaskTemplate` base class with `run_pipeline` orchestration
- [x] Pipeline modules: `TaxonomyExpander`, `GenerationEngine`, `CritiqueEngine`, `DedupEngine`, `HardNegativeMiner`
- [x] Task implementations: `ClassificationTask`, `ExtractionTask`, `InstructionTask`
- [x] Validation: `UnslothFinetuner`, `Evaluator`, `BenchmarkRunner`
- [x] Typer CLI with `generate`, `validate`, `benchmark` commands
- [x] Top-level `generate()` public API
- [x] Smoke-test examples: Banking-77, DocRED, Python instructions
- [x] Unit tests for models (no LLM calls)
- [x] README, VISION, benchmarks placeholder

**Blockers / Notes:**
- Unsloth validation requires a CUDA GPU — will run on Colab for benchmark numbers
- ChromaDB ephemeral client used for dedup (no persistence needed at generation time)
- Together AI uses the OpenAI-compatible endpoint; Groq routed by "versatile" substring

---

## Week 2 — First Benchmark Run

**Goals:**
- [ ] Run Banking-77 smoke test end-to-end (50 examples, check quality)
- [ ] Run DocRED smoke test
- [ ] Run Python instruction smoke test
- [ ] Set up Colab notebook for Unsloth fine-tuning
- [ ] Fill in benchmark/results.md with real numbers

---

## Week 3 — Quality Iteration

**Goals:**
- [ ] Analyse critique score distributions across task types
- [ ] Tune prompts based on low-scoring example patterns
- [ ] Add `--critique-threshold` CLI flag experiments
- [ ] Profile cost per example across providers

---

## Week 4 — Hard Negative + Dedup Tuning

**Goals:**
- [ ] Ablation: dedup threshold 0.85 vs 0.90 vs 0.95 impact on downstream F1
- [ ] Ablation: hard negative mining on/off impact on Banking-77 F1
- [ ] Document findings in benchmarks/results.md

---

## Backlog

- Multi-model orchestration (different generators + cross-model critique)
- HuggingFace dataset card auto-generation
- Dialogue task type
- Active learning loop
- Web UI prototype
