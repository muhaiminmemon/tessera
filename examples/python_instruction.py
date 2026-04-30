"""
Python instruction-following — grounded in real iamtarun/python_code_instructions_18k_alpaca
distribution. Top verbs: create (32%), write (22%), generate (11%), design (6%), develop (5%).
Mapped to 5 instruction types matching that distribution.
"""
from tessera import generate
import random


SPEC = {
    "domain": "Python programming",
    "instruction_types": [
        "create a Python function or class",      # maps to create (32%)
        "write a Python script or program",       # maps to write (22%)
        "generate Python code for a task",        # maps to generate (11%)
        "design a Python solution or algorithm",  # maps to design (6%)
        "develop a Python module or utility",     # maps to develop (5%)
    ],
    "response_format": "markdown with Python code blocks and brief explanation",
    "language": "English",
}


def main() -> None:
    print("=" * 60)
    print("Python Instruction-Following  |  Tessera benchmark")
    print("5 types grounded in real dataset verb distribution")
    print("=" * 60)

    result = generate(
        task="instruction",
        spec_dict=SPEC,
        n_examples=500,
        output_format="alpaca",
        output_path="outputs/python_instruction_tessera_train.jsonl",
    )

    print(f"\nPipeline stats:")
    print(f"  Generated:      {result.total_generated}")
    print(f"  After critique: {result.total_after_critique}")
    print(f"  After dedup:    {result.total_after_dedup}")
    print(f"  Final:          {len(result.examples)}")
    print(f"  Est. cost:     ${result.cost_usd:.4f}")

    print("\nSample instruction/response pairs:")
    for ex in random.sample(result.examples, min(3, len(result.examples))):
        print(f"\n  INSTRUCTION: {(ex.instruction or '')[:120]}")
        print(f"  RESPONSE:    {(ex.response or '')[:120]}...")

    print(f"\nOutput: outputs/python_instruction_tessera_train.jsonl")


if __name__ == "__main__":
    main()