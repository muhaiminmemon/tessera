"""
Unsloth LoRA fine-tuning for Tessera-generated datasets.
Run on Colab with GPU runtime (free T4 is sufficient for 3B models).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tessera.core.models import Example, TaskType

log = logging.getLogger(__name__)


def _examples_to_alpaca(examples: list[Example], task_type: TaskType) -> list[dict[str, str]]:
    rows = []
    for ex in examples:
        if task_type == TaskType.CLASSIFICATION:
            rows.append(
                {
                    "instruction": "Classify the following text into exactly one category.",
                    "input": ex.text or "",
                    "output": ex.label or "",
                }
            )
        elif task_type == TaskType.EXTRACTION:
            rows.append(
                {
                    "instruction": "Extract structured information from the text and return valid JSON.",
                    "input": ex.source_text or "",
                    "output": json.dumps(ex.extracted_fields or {}, ensure_ascii=False),
                }
            )
        else:  # INSTRUCTION
            rows.append(
                {
                    "instruction": ex.instruction or "",
                    "input": "",
                    "output": ex.response or "",
                }
            )
    return rows


_ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)


class UnslothFinetuner:
    def run(
        self,
        train_examples: list[Example],
        task_type: TaskType,
        output_dir: str,
        base_model: str = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length: int = 2048,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
    ) -> str:
        """Fine-tune model with Unsloth LoRA and save to output_dir. Returns path."""
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth is required for fine-tuning. "
                "Install with: pip install unsloth  (or use the [finetune] extra). "
                "Unsloth requires a CUDA GPU — run on Colab or a GPU instance."
            ) from e

        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        alpaca_rows = _examples_to_alpaca(train_examples, task_type)

        def format_row(row: dict[str, Any]) -> dict[str, str]:
            return {
                "text": _ALPACA_TEMPLATE.format(
                    instruction=row["instruction"],
                    input=row["input"],
                    output=row["output"],
                )
            }

        hf_dataset = Dataset.from_list([format_row(r) for r in alpaca_rows])

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                report_to="none",
            ),
        )

        trainer.train()
        model.save_pretrained(str(output_path / "lora_adapter"))
        tokenizer.save_pretrained(str(output_path / "lora_adapter"))

        saved_path = str(output_path / "lora_adapter")
        log.info("Model saved to %s", saved_path)
        return saved_path
