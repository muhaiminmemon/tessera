"""
DocRED relation extraction — grounded in actual DocRED top relation frequencies.
Top 8 relations by occurrence count across the training set (from research papers):
P17 country, P131 located_in_territory, P27 country_of_citizenship,
P150 contains_territory, P36 capital, P161 cast_member, P175 performer, P19 place_of_birth
"""
from tessera import generate
from collections import Counter


SPEC = {
    "domain": "Wikipedia biographical and geographical articles",
    "schema_definition": {
        "head_entity": "name of the subject entity (person, organization, or location)",
        "relation": (
            "one of exactly these values: "
            "country | located_in_territory | country_of_citizenship | "
            "contains_territory | capital | cast_member | performer | place_of_birth"
        ),
        "tail_entity": "name of the object entity (person, organization, or location)",
        "evidence_sentence": (
            "the single sentence from the source text that most directly "
            "expresses this relation between the two entities"
        ),
    },
    "source_text_type": "Wikipedia paragraph about a person, place, or organization",
    "language": "English",
}


def main() -> None:
    print("=" * 60)
    print("DocRED Relation Extraction  |  Tessera benchmark")
    print("8 relations grounded in real DocRED frequency distribution")
    print("=" * 60)

    result = generate(
        task="extraction",
        spec_dict=SPEC,
        n_examples=500,
        output_format="jsonl",
        output_path="outputs/docred_tessera_train.jsonl",
    )

    print(f"\nPipeline stats:")
    print(f"  Generated:      {result.total_generated}")
    print(f"  After critique: {result.total_after_critique}")
    print(f"  After dedup:    {result.total_after_dedup}")
    print(f"  Final:          {len(result.examples)}")
    print(f"  Est. cost:     ${result.cost_usd:.4f}")

    # Relation distribution
    rel_counts = Counter(
        ex.extracted_fields.get("relation", "unknown")
        for ex in result.examples
        if ex.extracted_fields
    )
    print("\nRelation distribution:")
    for rel, count in rel_counts.most_common():
        bar = "█" * count
        print(f"  {rel:<30} {count:>4}  {bar}")

    print("\nSample triples:")
    import random
    for ex in random.sample(result.examples, min(5, len(result.examples))):
        fields = ex.extracted_fields or {}
        print(
            f"  ({fields.get('head_entity')}, "
            f"{fields.get('relation')}, "
            f"{fields.get('tail_entity')})"
        )

    print(f"\nOutput: outputs/docred_tessera_train.jsonl")


if __name__ == "__main__":
    main()