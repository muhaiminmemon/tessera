from datasets import load_dataset
from collections import Counter

ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

print(f"Total examples: {len(ds)}")
print(f"Columns: {ds.column_names}")

print(f"\nFirst 3 instructions:")
for item in list(ds)[:3]:
    print(f"  - {item['instruction'][:120]}")

type_counts = Counter()
for item in ds:
    instr = item["instruction"].lower().strip()
    first_word = instr.split()[0] if instr else "unknown"
    type_counts[first_word] += 1

print(f"\nTop 20 opening verbs:")
for verb, count in type_counts.most_common(20):
    print(f"  {count:>5} | {verb}")
