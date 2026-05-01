"""
Enterprise RAG evaluation QA — generates (context, question, answer) triples
for evaluating RAG pipelines on enterprise documentation.

Uses factoid, multi-hop, and unanswerable question types only.
Abstractive is excluded because enterprise RAG requires grounded, extractable answers.
"""
from tessera import generate
import random
from collections import Counter


SPEC = {
    "domain": (
        "enterprise software documentation, internal company policies, "
        "and technical product manuals"
    ),
    "question_types": ["factoid", "multi-hop", "unanswerable"],
}


def main() -> None:
    print("=" * 60)
    print("Enterprise RAG Evaluation QA  |  Tessera benchmark")
    print("3 question types: factoid, multi-hop, unanswerable")
    print("=" * 60)

    result = generate(
        task="qa",
        spec_dict=SPEC,
        n_examples=150,
        output_format="jsonl",
        output_path="outputs/rag_eval_qa.jsonl",
    )

    print(f"\nPipeline stats:")
    print(f"  Generated:      {result.total_generated}")
    print(f"  After critique: {result.total_after_critique}")
    print(f"  After dedup:    {result.total_after_dedup}")
    print(f"  Final:          {len(result.examples)}")
    print(f"  Est. cost:     ${result.cost_usd:.4f}")

    type_counts = Counter(ex.question_type for ex in result.examples)
    diff_counts = Counter(ex.difficulty for ex in result.examples)

    print("\nQuestion type distribution:")
    for qt, count in sorted(type_counts.items()):
        bar = "█" * count
        print(f"  {qt:<20} {count:>4}  {bar}")

    print("\nDifficulty distribution:")
    for diff, count in sorted(diff_counts.items()):
        print(f"  {diff:<10} {count:>4}")

    print("\nSample QA triples:")
    for ex in random.sample(result.examples, min(4, len(result.examples))):
        print(f"\n  TYPE:     {ex.question_type}  [{ex.difficulty}]")
        print(f"  QUESTION: {(ex.question or '')[:120]}")
        print(f"  ANSWER:   {(ex.answer or '')[:120]}")

    print(f"\nOutput: outputs/rag_eval_qa.jsonl")


if __name__ == "__main__":
    main()
