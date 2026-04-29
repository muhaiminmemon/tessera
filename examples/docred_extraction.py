"""
DocRED-style relation extraction smoke test.

Generates 30 synthetic examples of entity-relation triples extracted from
news/Wikipedia-style text.  Prints extracted relation triples to stdout.
"""
from __future__ import annotations

from tessera import generate

SPEC = {
    "domain": "news articles and Wikipedia text",
    "schema_definition": {
        "subject": "entity name (person, org, or location)",
        "relation": "relation type, e.g. 'founded_by', 'located_in', 'works_for'",
        "object": "entity name (person, org, or location)",
        "sentence": "source sentence containing the relation",
    },
    "source_text_type": "news article paragraph",
    "language": "English",
}


def main() -> None:
    print("=" * 60)
    print("DocRED Relation Extraction  |  Tessera smoke test")
    print("=" * 60)

    result = generate(
        task="extraction",
        spec_dict=SPEC,
        n_examples=30,
        output_format="jsonl",
        output_path="outputs/docred_smoke_test.jsonl",
    )

    print(f"\nPipeline stats:")
    print(f"  Generated:      {result.total_generated}")
    print(f"  After critique: {result.total_after_critique}")
    print(f"  After dedup:    {result.total_after_dedup}")
    print(f"  Final:          {len(result.examples)}")
    print(f"  Est. cost:     ${result.cost_usd:.4f}")

    print("\nExtracted relation triples:")
    print(f"  {'#':<4} {'Subject':<20} {'Relation':<22} {'Object':<20}")
    print("  " + "-" * 68)
    for i, ex in enumerate(result.examples, 1):
        fields = ex.extracted_fields or {}
        subj = str(fields.get("subject", "?"))[:18]
        rel = str(fields.get("relation", "?"))[:20]
        obj = str(fields.get("object", "?"))[:18]
        print(f"  {i:<4} {subj:<20} {rel:<22} {obj:<20}")

    print("\nSample source texts:")
    for ex in result.examples[:3]:
        fields = ex.extracted_fields or {}
        print(f"\n  Source: {(ex.source_text or '')[:200]!r}")
        print(f"  Triple: ({fields.get('subject')}, {fields.get('relation')}, {fields.get('object')})")
        if ex.critique_scores:
            print(f"  Critique mean: {ex.critique_scores.mean:.1f}")

    print("\nOutput written to: outputs/docred_smoke_test.jsonl")


if __name__ == "__main__":
    main()
