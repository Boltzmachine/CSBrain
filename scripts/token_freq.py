from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm

# -----------------------
# Hardcoded defaults
# -----------------------
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en"
SPLIT = "train"
MAX_EXAMPLES = 1_000_000
OUTPUT_FILE = "llama2_token_frequencies.json"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print("Streaming dataset...")
dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split=SPLIT,
    streaming=True,
).shuffle(seed=42)

counter = Counter()
processed = 0

print("Counting tokens...")

for example in tqdm(dataset, total=MAX_EXAMPLES):
    if processed >= MAX_EXAMPLES:
        break

    text = example["text"]
    if not text.strip():
        continue

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    counter.update(token_ids)

    processed += 1

    if processed % 10000 == 0:
        print(f"Processed {processed} examples")

print("\nSaving to JSON...")

vocab_freq = []
for token_id in range(tokenizer.vocab_size):
    vocab_freq.append({
        "token_id": token_id,
        "token": tokenizer.decode([token_id]),
        "count": counter[token_id]
    })
# Save as: {token_id: count}
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab_freq, f, indent=4)

print("Done.")
print(f"Saved to {OUTPUT_FILE}")
print(f"Total tokens counted: {sum(counter.values())}")
print(f"Unique token IDs: {len(counter)}")