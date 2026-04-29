import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from tessera.core.models import Example, TaskType, ClassificationSpec
from tessera.pipeline.hard_negative import HardNegativeMiner

# Load examples
examples = []
with open("outputs/banking77_smoke_test.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line.strip())
        if not row:
            continue
        examples.append(Example(
            task_type=TaskType.CLASSIFICATION,
            text=row["text"],
            label=row["label"],
        ))

print(f"Loaded {len(examples)} examples")

spec = ClassificationSpec(
    domain="banking customer support",
    labels=["card_lost", "transfer_failed", "balance_inquiry",
            "account_locked", "foreign_transaction_declined"],
)

# Debug: inspect probability distribution
print("\nAnalyzing decision boundary confidence...")
texts = [ex.text or "" for ex in examples]
labels = [ex.label or "" for ex in examples]

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(texts, show_progress_bar=False)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(embeddings, labels)

probas = clf.predict_proba(embeddings)
max_probas = np.max(probas, axis=1)

print(f"Max probability stats:")
print(f"  Mean:       {max_probas.mean():.3f}")
print(f"  Min:        {max_probas.min():.3f}")
print(f"  Median:     {np.median(max_probas):.3f}")
print(f"  % below 0.6: {(max_probas < 0.60).mean():.1%}")
print(f"  % below 0.7: {(max_probas < 0.70).mean():.1%}")
print(f"  % below 0.8: {(max_probas < 0.80).mean():.1%}")
print(f"  % below 0.9: {(max_probas < 0.90).mean():.1%}")

# Pick threshold at the 20th percentile so we always find some hard negatives
threshold = float(np.percentile(max_probas, 20))
print(f"\nUsing adaptive threshold (20th percentile): {threshold:.3f}")

# Find hard negatives manually so we can inspect them
hard_indices = [i for i, p in enumerate(max_probas) if p < threshold]
print(f"Hard negatives found: {len(hard_indices)}")

print("\nSample hard negatives (most ambiguous examples):")
# Sort by confidence ascending — least confident first
hard_indices_sorted = sorted(hard_indices, key=lambda i: max_probas[i])
for idx in hard_indices_sorted[:10]:
    ex = examples[idx]
    proba_row = probas[idx]
    label_probs = dict(zip(clf.classes_, proba_row))
    top2 = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"\n  label={ex.label} | confidence={max_probas[idx]:.3f}")
    print(f"  top2 predictions: {top2[0][0]}={top2[0][1]:.2f}, {top2[1][0]}={top2[1][1]:.2f}")
    print(f"  text: {(ex.text or '')[:120]}")

# Now run the actual miner with adaptive threshold
print("\nRunning HardNegativeMiner...")

class AdaptiveHardNegativeMiner(HardNegativeMiner):
    def _mine_sklearn(self, examples, oversample_factor):
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        import copy
        import uuid

        texts = [ex.text or "" for ex in examples]
        labels = [ex.label or "" for ex in examples]

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = encoder.encode(texts, show_progress_bar=False)

        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return examples

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(embeddings, labels)

        probas = clf.predict_proba(embeddings)
        max_proba = np.max(probas, axis=1)

        # Adaptive: use 20th percentile instead of hardcoded 0.6
        threshold = float(np.percentile(max_proba, 20))
        hard_mask = max_proba < threshold
        hard_indices = [i for i, is_hard in enumerate(hard_mask) if is_hard]

        if not hard_indices:
            return examples

        target_total = int(len(examples) * oversample_factor)
        hard_negatives = [examples[i] for i in hard_indices]

        result = list(examples)
        added = 0
        max_to_add = target_total - len(examples)

        while added < max_to_add and hard_negatives:
            for ex in hard_negatives:
                if added >= max_to_add:
                    break
                dup = ex.model_copy(update={"id": str(uuid.uuid4())})
                result.append(dup)
                added += 1

        return result

miner = AdaptiveHardNegativeMiner()
result = miner.mine(examples, spec, oversample_factor=1.5)

original_count = len(examples)
hard_negative_count = len(result) - original_count

print(f"\nOriginal examples:      {original_count}")
print(f"After hard neg mining:  {len(result)}")
print(f"Hard negatives added:   {hard_negative_count}")

# Save augmented dataset
output_path = "outputs/banking77_with_hard_negatives.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for ex in result:
        f.write(json.dumps({"text": ex.text, "label": ex.label}, ensure_ascii=False) + "\n")

print(f"\nAugmented dataset saved to: {output_path}")
print(f"This is your final training set for the Banking77 validation experiment.")