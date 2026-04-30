import urllib.request
import json
from collections import Counter

# Correct raw URL for DocRED data
url = "https://raw.githubusercontent.com/thunlp/DocRED/master/data/train_annotated.json"

print("Downloading DocRED rel_info.json...")
rel_url = "https://raw.githubusercontent.com/thunlp/DocRED/master/meta/rel_info.json"

try:
    urllib.request.urlretrieve(rel_url, "scripts/docred_rel_info.json")
    print("rel_info downloaded")
except Exception as e:
    print(f"rel_info failed: {e}")

print("Downloading train_annotated.json...")
try:
    urllib.request.urlretrieve(url, "scripts/docred_train.json")
    print("train data downloaded")
except Exception as e:
    print(f"train data failed: {e}")
    # Fallback — just use hardcoded top relations from papers
    print("\nUsing hardcoded top relations from research papers:")
    top_relations = [
        ("P17",  "country"),
        ("P131", "located in the administrative territorial entity"),
        ("P27",  "country of citizenship"),
        ("P150", "contains administrative territorial entity"),
        ("P36",  "capital"),
        ("P161", "cast member"),
        ("P175", "performer"),
        ("P178", "developer"),
        ("P19",  "place of birth"),
        ("P20",  "place of death"),
        ("P22",  "father"),
        ("P25",  "mother"),
        ("P26",  "spouse"),
        ("P40",  "child"),
        ("P58",  "screenwriter"),
    ]
    for pid, name in top_relations:
        print(f"  {pid:<8} | {name}")
    exit()

with open("scripts/docred_train.json") as f:
    data = json.load(f)

with open("scripts/docred_rel_info.json") as f:
    rel_info = json.load(f)

print(f"Total documents: {len(data)}")

relation_counts = Counter()
for doc in data:
    for label in doc.get("labels", []):
        relation_counts[label["r"]] += 1

print(f"\nTop 15 most frequent relations:")
print(f"  {'Count':>7} | {'ID':<8} | Name")
print(f"  {'-'*50}")
for rel_id, count in relation_counts.most_common(15):
    name = rel_info.get(rel_id, "unknown")
    print(f"  {count:>7} | {rel_id:<8} | {name}")