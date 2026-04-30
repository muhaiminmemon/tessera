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
    "lost_or_stolen_card",
    "failed_transfer",
    "balance_not_updated_after_bank_transfer",
    "pin_blocked",
    "card_payment_wrong_exchange_rate",
]

LABEL_DESCRIPTIONS = {
    "lost_or_stolen_card": "Customer reporting their debit or credit card is lost or stolen",
    "failed_transfer": "Customer reporting that a bank transfer failed or did not complete",
    "balance_not_updated_after_bank_transfer": "Customer saying their balance has not updated after a bank transfer",
    "pin_blocked": "Customer saying their card PIN is blocked or they cannot use it due to too many failed PIN attempts",
    "card_payment_wrong_exchange_rate": "Customer saying a card payment used the wrong exchange rate for a foreign currency transaction",
}

SPEC = {
    "domain": "retail banking customer support",
    "labels": LABELS,
    "label_descriptions": LABEL_DESCRIPTIONS,
    "language": "English",
    "example_inputs": [
        "I lost my card and need to block it",
        "My bank transfer failed",
        "My balance has not updated after my transfer",
        "My PIN is blocked after too many attempts",
        "My foreign card payment used the wrong exchange rate",
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
