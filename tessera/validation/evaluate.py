"""Evaluate a fine-tuned model against a held-out test set."""
from __future__ import annotations

import json
from typing import Any

from tessera.core.models import Example, TaskType, ValidationMetrics


class Evaluator:
    def evaluate(
        self,
        model_path: str,
        test_examples: list[Example],
        task_type: TaskType,
        base_model: str = "unsloth/Llama-3.2-3B-Instruct",
        judge_model: str = "gpt-4o-mini",
        dataset_name: str = "tessera",
    ) -> ValidationMetrics:
        if task_type == TaskType.CLASSIFICATION:
            return self._evaluate_classification(
                model_path, test_examples, base_model, dataset_name
            )
        elif task_type == TaskType.EXTRACTION:
            return self._evaluate_extraction(
                model_path, test_examples, base_model, dataset_name
            )
        else:  # INSTRUCTION
            return self._evaluate_instruction(
                model_path, test_examples, judge_model, dataset_name
            )

    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, base_model: str) -> tuple[Any, Any]:
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth required for model evaluation. "
                "Install: pip install unsloth"
            ) from e

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _generate(self, model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 128) -> str:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt prefix
        return decoded[len(prompt):].strip()

    def _evaluate_classification(
        self,
        model_path: str,
        test_examples: list[Example],
        base_model: str,
        dataset_name: str,
    ) -> ValidationMetrics:
        from sklearn.metrics import f1_score, accuracy_score

        model, tokenizer = self._load_model(model_path, base_model)

        y_true, y_pred = [], []
        for ex in test_examples:
            prompt = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\nClassify the following text into exactly one category.\n\n"
                f"### Input:\n{ex.text}\n\n### Response:\n"
            )
            pred = self._generate(model, tokenizer, prompt, max_new_tokens=32)
            y_true.append(ex.label or "")
            y_pred.append(pred.strip())

        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        return ValidationMetrics(
            task_type=TaskType.CLASSIFICATION,
            model=model_path,
            dataset=dataset_name,
            n_train=0,
            n_test=len(test_examples),
            f1_macro=f1,
            accuracy=acc,
        )

    def _evaluate_extraction(
        self,
        model_path: str,
        test_examples: list[Example],
        base_model: str,
        dataset_name: str,
    ) -> ValidationMetrics:
        from sklearn.metrics import f1_score

        model, tokenizer = self._load_model(model_path, base_model)

        per_field_matches: dict[str, list[int]] = {}
        valid_json_count = 0

        for ex in test_examples:
            prompt = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\nExtract structured information from the text and return valid JSON.\n\n"
                f"### Input:\n{ex.source_text}\n\n### Response:\n"
            )
            pred_str = self._generate(model, tokenizer, prompt, max_new_tokens=512)

            try:
                pred_fields = json.loads(pred_str)
                valid_json_count += 1
            except json.JSONDecodeError:
                pred_fields = {}

            gold_fields = ex.extracted_fields or {}
            for key, gold_val in gold_fields.items():
                pred_val = pred_fields.get(key, "")
                match = 1 if str(pred_val).strip() == str(gold_val).strip() else 0
                per_field_matches.setdefault(key, []).append(match)

        per_field_f1 = {
            k: float(sum(v)) / len(v) if v else 0.0
            for k, v in per_field_matches.items()
        }
        overall_f1 = float(sum(per_field_f1.values()) / len(per_field_f1)) if per_field_f1 else 0.0
        validity_rate = valid_json_count / len(test_examples) if test_examples else 0.0

        return ValidationMetrics(
            task_type=TaskType.EXTRACTION,
            model=model_path,
            dataset=dataset_name,
            n_train=0,
            n_test=len(test_examples),
            f1_macro=overall_f1,
            per_field_f1=per_field_f1,
            json_validity_rate=validity_rate,
        )

    def _evaluate_instruction(
        self,
        model_path: str,
        test_examples: list[Example],
        judge_model: str,
        dataset_name: str,
    ) -> ValidationMetrics:
        from tessera.core.llm_client import get_client

        client = get_client()
        scores: list[float] = []

        for ex in test_examples:
            system = (
                "You are an impartial judge evaluating AI assistant responses. "
                "Score the response 0-10 based on correctness, helpfulness, and quality. "
                "Return ONLY a JSON object: {\"score\": <0-10>}"
            )
            user = (
                f"Instruction: {ex.instruction}\n\n"
                f"Response to evaluate: {ex.response}\n\n"
                "Score this response 0-10."
            )
            try:
                raw = client.complete(
                    model=judge_model,
                    system=system,
                    user=user,
                    temperature=0.0,
                    max_tokens=64,
                    json_mode=True,
                )
                data = json.loads(raw)
                scores.append(float(data.get("score", 5.0)))
            except Exception:
                scores.append(5.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return ValidationMetrics(
            task_type=TaskType.INSTRUCTION,
            model=model_path,
            dataset=dataset_name,
            n_train=0,
            n_test=len(test_examples),
            llm_judge_score=avg_score,
        )
