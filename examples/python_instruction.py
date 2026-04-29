"""
Python programming instruction-following smoke test.

Generates 30 synthetic instruction/response pairs covering four Python task types.
Prints sample pairs to stdout.
"""
from __future__ import annotations

from tessera import generate

SPEC = {
    "domain": "Python programming",
    "instruction_types": [
        "write a Python function",
        "debug Python code",
        "explain Python concept",
        "refactor Python code",
    ],
    "response_format": "markdown with code blocks",
    "language": "English",
}


def main() -> None:
    print("=" * 60)
    print("Python Instruction-Following  |  Tessera smoke test")
    print("=" * 60)

    result = generate(
        task="instruction",
        spec_dict=SPEC,
        n_examples=30,
        output_format="alpaca",
        output_path="outputs/python_instruction_smoke_test.jsonl",
    )

    print(f"\nPipeline stats:")
    print(f"  Generated:      {result.total_generated}")
    print(f"  After critique: {result.total_after_critique}")
    print(f"  After dedup:    {result.total_after_dedup}")
    print(f"  Final:          {len(result.examples)}")
    print(f"  Est. cost:     ${result.cost_usd:.4f}")

    # Count instruction types (via taxonomy_node_id not available directly, use first word)
    from collections import Counter

    def infer_type(instruction: str) -> str:
        instruction_lower = (instruction or "").lower()
        for itype in SPEC["instruction_types"]:
            if itype.split()[0] in instruction_lower or itype.split()[-1] in instruction_lower:
                return itype
        return "other"

    type_counts = Counter(infer_type(ex.instruction) for ex in result.examples)
    print("\nInstruction type distribution:")
    for itype, count in sorted(type_counts.items()):
        print(f"  {itype:<35} {count}")

    print("\n" + "=" * 60)
    print("Sample instruction/response pairs")
    print("=" * 60)

    for i, ex in enumerate(result.examples[:5], 1):
        print(f"\n--- Example {i} ---")
        print(f"INSTRUCTION:\n{ex.instruction}")
        print(f"\nRESPONSE:\n{(ex.response or '')[:400]}")
        if len(ex.response or "") > 400:
            print("  [truncated...]")
        if ex.critique_scores:
            print(
                f"\n  Critique: mean={ex.critique_scores.mean:.1f}  "
                f"clarity={ex.critique_scores.realism:.0f}  "
                f"quality={ex.critique_scores.label_correctness:.0f}"
            )

    print("\nOutput written to: outputs/python_instruction_smoke_test.jsonl")


if __name__ == "__main__":
    main()
