"""
Banking-77 classification smoke test.

Generates synthetic banking support examples across 5 intent labels,
prints pipeline stats, label distribution, and 5 random samples.
Writes output to outputs/banking77_smoke_test.jsonl.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from tessera import generate
from tessera.core.models import TaskType

LABELS = [
    "card_lost",
    "transfer_failed",
    "balance_inquiry",
    "account_locked",
    "foreign_transaction_declined",
]

LABEL_DESCRIPTIONS = {
    "card_lost": "Customer reporting their debit or credit card is lost or stolen",
    "transfer_failed": "Customer reporting a failed or pending money transfer",
    "balance_inquiry": "Customer asking about their account balance or recent transactions",
    "account_locked": "Customer unable to log in or whose account has been locked/suspended",
    "foreign_transaction_declined": "Customer whose card was declined while abroad or on a foreign purchase",
}

SPEC = {
    "domain": "retail banking customer support",
    "labels": LABELS,
    "label_descriptions": LABEL_DESCRIPTIONS,
    "language": "English",
    "example_inputs": [
        "I can't find my card anywhere",
        "My payment didn't go through",
        "What's my current balance?",
    ],
}


def main() -> None:
    print("=" * 60)
    print("Banking-77 Classification  |  Tessera smoke test")
    print("=" * 60)

    result = generate(
        task="classification",
        spec_dict=SPEC,
        n_examples=1000,
        output_format="jsonl",
        output_path="outputs/banking77_smoke_test.jsonl",
    )

    # Pipeline stats
    print(f"\nPipeline stats:")
    print(f"  Generated:        {result.total_generated}")
    print(f"  After critique:   {result.total_after_critique}")
    print(f"  After dedup:      {result.total_after_dedup}")
    print(f"  Final examples:   {len(result.examples)}")
    print(f"  Est. cost:       ${result.cost_usd:.4f}")

    # Label distribution bar chart
    label_counts = Counter(ex.label for ex in result.examples)
    print("\nLabel distribution:")
    max_count = max(label_counts.values(), default=1)
    bar_width = 30
    for label in LABELS:
        count = label_counts.get(label, 0)
        bar = "█" * int(count / max_count * bar_width)
        print(f"  {label:<35} {bar} {count}")

    # 5 random samples with critique scores
    samples = random.sample(result.examples, min(5, len(result.examples)))
    print("\n5 random samples:")
    for i, ex in enumerate(samples, 1):
        score_str = ""
        if ex.critique_scores:
            score_str = (
                f"  [critique mean={ex.critique_scores.mean:.1f}  "
                f"realism={ex.critique_scores.realism:.0f}  "
                f"correctness={ex.critique_scores.label_correctness:.0f}  "
                f"specificity={ex.critique_scores.specificity:.0f}]"
            )
        print(f"\n  [{i}] label={ex.label}")
        print(f"      text={ex.text!r:.120}")
        if score_str:
            print(score_str)

    output_path = Path("outputs/banking77_smoke_test.jsonl")
    print(f"\nOutput written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
