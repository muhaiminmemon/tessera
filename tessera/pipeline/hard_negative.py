"""Hard-negative miner for classification tasks."""
from __future__ import annotations

import warnings

from tessera.core.models import ClassificationSpec, Example, TaskType


class HardNegativeMiner:
    """
    Finds examples near classifier decision boundaries and duplicates them
    to increase training difficulty.  Classification-only.
    """

    def mine(
        self,
        examples: list[Example],
        spec: ClassificationSpec,
        oversample_factor: float = 2.0,
    ) -> list[Example]:
        if not examples:
            return examples

        for ex in examples:
            if ex.task_type != TaskType.CLASSIFICATION:
                raise ValueError("HardNegativeMiner only supports CLASSIFICATION examples")

        try:
            return self._mine_sklearn(examples, oversample_factor)
        except ImportError as exc:
            warnings.warn(
                f"[HardNegativeMiner] required dependency missing ({exc}); "
                "returning original examples unchanged."
            )
            return examples

    def _mine_sklearn(
        self,
        examples: list[Example],
        oversample_factor: float,
    ) -> list[Example]:
        import uuid

        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression

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

        # Examples where the classifier is uncertain (near decision boundary)
        hard_mask = max_proba < 0.6
        hard_indices = [i for i, is_hard in enumerate(hard_mask) if is_hard]

        if not hard_indices:
            return examples

        # Duplicate hard negatives up to oversample_factor * original count
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
