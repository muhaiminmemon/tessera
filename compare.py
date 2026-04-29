from datasets import load_dataset

ds = load_dataset("legacy-datasets/banking77")

# Filter to show only labels that match your 5 intents
# Banking77 label IDs for your intents:
# card_lost = 42, balance_inquiry = 7, account_locked = 0
target_labels = [0, 7, 42]

print("--- Real Banking77 examples (your labels) ---")
count = 0
for item in ds["train"]:
    if item["label"] in target_labels and count < 15:
        print(f"label={item['label']} | {item['text']}")
        count += 1