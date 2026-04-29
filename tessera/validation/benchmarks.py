"""Benchmark runner: compares Tessera-trained vs real-data vs random baseline."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tessera.core.models import Example, TaskType
from tessera.validation.evaluate import Evaluator
from tessera.validation.finetune import UnslothFinetuner


class BenchmarkRunner:
    def run_experiment(
        self,
        task_type: TaskType,
        tessera_train: list[Example],
        real_train: list[Example],
        test: list[Example],
        output_dir: str,
        base_model: str = "unsloth/Llama-3.2-3B-Instruct",
        judge_model: str = "gpt-4o-mini",
    ) -> dict[str, Any]:
        """
        Run three training conditions and return comparative metrics.

        Conditions:
          1. tessera_trained  — fine-tuned on Tessera synthetic data
          2. real_data_trained — fine-tuned on real labelled data
          3. random_baseline  — no training; labels randomly shuffled

        Returns dict with f1 scores for all three conditions plus pct_of_real.
        """
        finetuner = UnslothFinetuner()
        evaluator = Evaluator()

        out = Path(output_dir)

        # --- Condition 1: Tessera ---
        tessera_model_path = finetuner.run(
            train_examples=tessera_train,
            task_type=task_type,
            output_dir=str(out / "tessera_model"),
            base_model=base_model,
        )
        tessera_metrics = evaluator.evaluate(
            model_path=tessera_model_path,
            test_examples=test,
            task_type=task_type,
            base_model=base_model,
            judge_model=judge_model,
            dataset_name="tessera",
        )

        # --- Condition 2: Real data ---
        real_model_path = finetuner.run(
            train_examples=real_train,
            task_type=task_type,
            output_dir=str(out / "real_model"),
            base_model=base_model,
        )
        real_metrics = evaluator.evaluate(
            model_path=real_model_path,
            test_examples=test,
            task_type=task_type,
            base_model=base_model,
            judge_model=judge_model,
            dataset_name="real_data",
        )

        # --- Condition 3: Random baseline (no fine-tuning needed) ---
        random_f1 = self._random_baseline(test, task_type)

        tessera_f1 = tessera_metrics.f1_macro or tessera_metrics.llm_judge_score
        real_f1 = real_metrics.f1_macro or real_metrics.llm_judge_score

        pct_of_real = (tessera_f1 / real_f1 * 100) if real_f1 > 0 else 0.0

        return {
            "tessera_f1": tessera_f1,
            "real_data_f1": real_f1,
            "random_f1": random_f1,
            "pct_of_real": pct_of_real,
            "tessera_metrics": tessera_metrics.model_dump(),
            "real_metrics": real_metrics.model_dump(),
            "n_tessera_train": len(tessera_train),
            "n_real_train": len(real_train),
            "n_test": len(test),
        }

    @staticmethod
    def _random_baseline(test: list[Example], task_type: TaskType) -> float:
        if task_type != TaskType.CLASSIFICATION:
            return 0.0

        try:
            from sklearn.metrics import f1_score
        except ImportError:
            return 0.0

        labels = [ex.label for ex in test if ex.label]
        unique = list(set(labels))
        if not unique:
            return 0.0

        shuffled = [random.choice(unique) for _ in labels]
        return float(f1_score(labels, shuffled, average="macro", zero_division=0))
