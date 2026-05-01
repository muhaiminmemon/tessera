"""Tessera CLI — generate, validate, benchmark."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="tessera",
    help="Multi-task synthetic data framework for fine-tuning small LLMs.",
    add_completion=False,
)
console = Console()


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.command()
def generate(
    task: str = typer.Option(..., help="Task type: classification | extraction | instruction"),
    domain: str = typer.Option(..., help="Domain description, e.g. 'banking customer support'"),
    labels: Optional[str] = typer.Option(
        None, help="Comma-separated labels (classification only)"
    ),
    n: int = typer.Option(1000, help="Number of examples to generate"),
    output: str = typer.Option("dataset.jsonl", help="Output file path"),
    model: str = typer.Option(
        None, help="LLM model to use (default: TESSERA_DEFAULT_MODEL env var or gpt-4o-mini)"
    ),
    fmt: str = typer.Option("jsonl", help="Output format: jsonl | alpaca | sharegpt"),
) -> None:
    """Generate a synthetic dataset for fine-tuning."""
    from tessera import generate as tessera_generate
    from tessera.core.models import (
        ClassificationSpec,
        ExtractionSpec,
        InstructionSpec,
        TaskType,
    )

    model = model or os.environ.get("TESSERA_DEFAULT_MODEL", "gpt-4o-mini")
    task_type = TaskType(task.lower())

    console.print(f"[bold green]Tessera[/bold green] — generating [cyan]{n}[/cyan] examples")
    console.print(f"  Task:   {task_type.value}")
    console.print(f"  Domain: {domain}")
    console.print(f"  Model:  {model}")

    if task_type == TaskType.CLASSIFICATION:
        if not labels:
            console.print("[red]--labels is required for classification tasks[/red]")
            raise typer.Exit(1)
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        spec = ClassificationSpec(domain=domain, labels=label_list)
    elif task_type == TaskType.EXTRACTION:
        console.print(
            "[yellow]No --labels needed for extraction. Using default schema.[/yellow]"
        )
        spec = ExtractionSpec(
            domain=domain,
            schema_definition={"field_1": "description", "field_2": "description"},
        )
    elif task_type == TaskType.QA:
        from tessera.core.models import QASpec
        spec = QASpec(domain=domain)
    else:  # INSTRUCTION
        spec = InstructionSpec(
            domain=domain,
            instruction_types=["explain", "write", "debug", "review"],
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running pipeline...", total=None)
        result = tessera_generate(
            task=task_type.value,
            spec_dict=spec.model_dump(),
            n_examples=n,
            model=model,
            output_format=fmt,
            output_path=output,
        )

    console.print(f"\n[bold]Pipeline summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("Count", justify="right")
    table.add_row("Generated", str(result.total_generated))
    table.add_row("After critique", str(result.total_after_critique))
    table.add_row("After dedup", str(result.total_after_dedup))
    table.add_row("Final output", str(len(result.examples)))
    table.add_row("Est. cost (USD)", f"${result.cost_usd:.4f}")
    console.print(table)
    console.print(f"\n[green]Saved to[/green] {output}")


@app.command()
def validate(
    task: str = typer.Option(..., help="Task type: classification | extraction | instruction"),
    train: str = typer.Option(..., help="Path to training JSONL"),
    test: str = typer.Option(..., help="Path to test JSONL"),
    output_dir: str = typer.Option("./validation_output", help="Directory to save model"),
    base_model: str = typer.Option(
        "unsloth/Llama-3.2-3B-Instruct", help="Base model to fine-tune"
    ),
) -> None:
    """Fine-tune a model on training data and evaluate on test data."""
    from tessera.core.models import Example, TaskType
    from tessera.validation.finetune import UnslothFinetuner
    from tessera.validation.evaluate import Evaluator

    task_type = TaskType(task.lower())

    console.print("[bold green]Tessera validate[/bold green]")
    console.print(f"  Task:       {task_type.value}")
    console.print(f"  Train:      {train}")
    console.print(f"  Test:       {test}")
    console.print(f"  Output dir: {output_dir}")

    train_rows = _load_jsonl(train)
    test_rows = _load_jsonl(test)

    def rows_to_examples(rows: list[dict], tt: TaskType) -> list[Example]:
        examples = []
        for r in rows:
            try:
                examples.append(Example(task_type=tt, **r))
            except Exception:
                pass
        return examples

    train_examples = rows_to_examples(train_rows, task_type)
    test_examples = rows_to_examples(test_rows, task_type)

    console.print(f"  Train examples: {len(train_examples)}")
    console.print(f"  Test examples:  {len(test_examples)}")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        p.add_task("Fine-tuning...", total=None)
        finetuner = UnslothFinetuner()
        model_path = finetuner.run(
            train_examples=train_examples,
            task_type=task_type,
            output_dir=output_dir,
            base_model=base_model,
        )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        p.add_task("Evaluating...", total=None)
        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            model_path=model_path,
            test_examples=test_examples,
            task_type=task_type,
            base_model=base_model,
        )

    console.print("\n[bold]Evaluation results[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("F1 (macro)", f"{metrics.f1_macro:.4f}")
    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    if metrics.llm_judge_score:
        table.add_row("LLM judge score", f"{metrics.llm_judge_score:.2f}/10")
    if metrics.json_validity_rate:
        table.add_row("JSON validity", f"{metrics.json_validity_rate:.2%}")
    console.print(table)


@app.command()
def benchmark(
    task: str = typer.Option(..., help="Task type: classification | extraction | instruction"),
    tessera_train: str = typer.Option(..., help="Path to Tessera-generated training JSONL"),
    real_train: str = typer.Option(..., help="Path to real training data JSONL"),
    test: str = typer.Option(..., help="Path to test JSONL"),
    output_dir: str = typer.Option("./benchmark_output", help="Directory for saved models"),
    base_model: str = typer.Option("unsloth/Llama-3.2-3B-Instruct", help="Base model"),
) -> None:
    """Compare Tessera-trained vs real-data-trained vs random baseline."""
    from tessera.core.models import Example, TaskType
    from tessera.validation.benchmarks import BenchmarkRunner

    task_type = TaskType(task.lower())

    console.print("[bold green]Tessera benchmark[/bold green]")

    def load_examples(path: str, tt: TaskType) -> list[Example]:
        rows = _load_jsonl(path)
        examples = []
        for r in rows:
            try:
                examples.append(Example(task_type=tt, **r))
            except Exception:
                pass
        return examples

    tessera_examples = load_examples(tessera_train, task_type)
    real_examples = load_examples(real_train, task_type)
    test_examples = load_examples(test, task_type)

    console.print(f"  Tessera train: {len(tessera_examples)} examples")
    console.print(f"  Real train:    {len(real_examples)} examples")
    console.print(f"  Test:          {len(test_examples)} examples")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        p.add_task("Running benchmark experiment...", total=None)
        runner = BenchmarkRunner()
        results = runner.run_experiment(
            task_type=task_type,
            tessera_train=tessera_examples,
            real_train=real_examples,
            test=test_examples,
            output_dir=output_dir,
            base_model=base_model,
        )

    console.print("\n[bold]Benchmark results[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Condition")
    table.add_column("F1 / Score", justify="right")
    table.add_row("Tessera synthetic", f"{results['tessera_f1']:.4f}")
    table.add_row("Real data", f"{results['real_data_f1']:.4f}")
    table.add_row("Random baseline", f"{results['random_f1']:.4f}")
    table.add_row("Tessera % of real data", f"{results['pct_of_real']:.1f}%")
    console.print(table)


@app.command()
def qa(
    domain: str = typer.Option(..., help="Domain for QA generation"),
    n_examples: int = typer.Option(500, help="Number of QA pairs to generate"),
    output_path: str = typer.Option("qa_dataset.jsonl", help="Output file path"),
    model: Optional[str] = typer.Option(
        None, help="LLM model (default: TESSERA_DEFAULT_MODEL env var or gpt-4o-mini)"
    ),
    fmt: str = typer.Option("jsonl", help="Output format: jsonl | squad | alpaca"),
    question_types: Optional[str] = typer.Option(
        None,
        help="Comma-separated question types (default: factoid,multi-hop,abstractive,unanswerable)",
    ),
) -> None:
    """Generate a synthetic QA dataset for RAG evaluation or SQuAD-style fine-tuning."""
    from tessera import generate as tessera_generate

    model = model or os.environ.get("TESSERA_DEFAULT_MODEL", "gpt-4o-mini")
    spec_dict: dict = {"domain": domain}
    if question_types:
        spec_dict["question_types"] = [q.strip() for q in question_types.split(",") if q.strip()]

    console.print(
        f"[bold green]Tessera QA[/bold green] — generating [cyan]{n_examples}[/cyan] QA pairs"
    )
    console.print(f"  Domain: {domain}")
    console.print(f"  Model:  {model}")
    if question_types:
        console.print(f"  Types:  {question_types}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running pipeline...", total=None)
        result = tessera_generate(
            task="qa",
            spec_dict=spec_dict,
            n_examples=n_examples,
            model=model,
            output_format=fmt,
            output_path=output_path,
        )

    console.print("\n[bold]Pipeline summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage")
    table.add_column("Count", justify="right")
    table.add_row("Generated", str(result.total_generated))
    table.add_row("After critique", str(result.total_after_critique))
    table.add_row("After dedup", str(result.total_after_dedup))
    table.add_row("Final output", str(len(result.examples)))
    table.add_row("Est. cost (USD)", f"${result.cost_usd:.4f}")
    console.print(table)
    console.print(f"\n[green]Saved to[/green] {output_path}")


if __name__ == "__main__":
    app()
