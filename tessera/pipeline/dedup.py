"""Deduplication engine using sentence-transformers + ChromaDB."""
from __future__ import annotations

import logging
import uuid
import warnings

from tessera.core.models import Example, TaskType

log = logging.getLogger(__name__)


class DedupEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._encoder = None

    def _get_encoder(self) -> object:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _example_text(self, ex: Example) -> str:
        if ex.task_type == TaskType.CLASSIFICATION:
            return ex.text or ""
        elif ex.task_type == TaskType.EXTRACTION:
            return ex.source_text or ""
        elif ex.task_type == TaskType.QA:
            # Embed only the question — context is always unique per example so
            # including it dilutes similarity and misses near-duplicate questions
            # that differ only in fictional product/company names.
            return ex.question or ""
        else:  # INSTRUCTION
            return (ex.instruction or "") + " " + (ex.response or "")

    def deduplicate(
        self,
        examples: list[Example],
        threshold: float = 0.90,
    ) -> list[Example]:
        if len(examples) <= 1:
            return examples

        try:
            result = self._dedup_chromadb(examples, threshold)
            log.info("dedup: %d → %d examples (threshold=%.2f)", len(examples), len(result), threshold)
            return result
        except ImportError:
            warnings.warn(
                "[DedupEngine] chromadb not installed; falling back to no dedup. "
                "Install it with: pip install chromadb"
            )
            return examples

    def _dedup_chromadb(
        self,
        examples: list[Example],
        threshold: float,
    ) -> list[Example]:
        import chromadb

        encoder = self._get_encoder()
        texts = [self._example_text(ex) for ex in examples]
        embeddings = encoder.encode(texts, show_progress_bar=False).tolist()

        client = chromadb.Client()
        collection_name = f"tessera_dedup_{uuid.uuid4().hex[:8]}"
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        kept: list[Example] = []
        kept_ids: list[str] = []
        kept_embeddings: list[list[float]] = []

        for i, (ex, emb) in enumerate(zip(examples, embeddings)):
            if not kept_ids:
                kept.append(ex)
                kept_ids.append(ex.id)
                kept_embeddings.append(emb)
                collection.add(ids=[ex.id], embeddings=[emb])
                continue

            results = collection.query(
                query_embeddings=[emb],
                n_results=1,
                include=["distances"],
            )
            distances = results["distances"][0]
            if distances:
                # ChromaDB cosine distance: similarity = 1 - distance
                similarity = 1.0 - distances[0]
                if similarity >= threshold:
                    continue  # near-duplicate, skip

            kept.append(ex)
            kept_ids.append(ex.id)
            collection.add(ids=[ex.id], embeddings=[emb])

        # Clean up ephemeral collection
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        return kept
