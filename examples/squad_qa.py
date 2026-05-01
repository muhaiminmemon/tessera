"""
SQuAD-style QA — generates a balanced (context, question, answer) dataset
for benchmarking against SQuAD F1. All four question types, ~125 each.
"""
from tessera import generate
import random


SPEC = {
    "domain": (
        "general knowledge encyclopedia articles covering science, history, "
        "geography, and technology"
    ),
    "question_types": ["factoid", "multi-hop", "abstractive", "unanswerable"],
}


def main() -> None:
    print("=" * 60)
    print("SQuAD-style QA  |  Tessera benchmark")
    print("4 question types × ~125 examples each")
    print("=" * 60)

    result = generate(
        task="qa",
        spec_dict=SPEC,
        n_examples=150,
        output_format="squad",
        output_path="outputs/squad_style_qa.jsonl",
    )

    print(f"\nPipeline stats:")
    print(f"  Generated:      {result.total_generated}")
    print(f"  After critique: {result.total_after_critique}")
    print(f"  After dedup:    {result.total_after_dedup}")
    print(f"  Final:          {len(result.examples)}")
    print(f"  Est. cost:     ${result.cost_usd:.4f}")

    from collections import Counter
    type_counts = Counter(ex.question_type for ex in result.examples)
    print("\nQuestion type distribution:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt:<20} {count:>4}")

    print("\nSample QA triples:")
    for ex in random.sample(result.examples, min(3, len(result.examples))):
        print(f"\n  TYPE:     {ex.question_type}  [{ex.difficulty}]")
        print(f"  QUESTION: {(ex.question or '')[:100]}")
        print(f"  ANSWER:   {(ex.answer or '')[:100]}")

    print(f"\nOutput: outputs/squad_style_qa.jsonl")


if __name__ == "__main__":
    main()
