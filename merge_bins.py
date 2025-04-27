import os
import json
import numpy as np

# Input dataset folders
dataset1 = "final_tokenized_bin"
dataset2 = "tokenized_chatml_bin"
output_dir = "merged_dataset"
os.makedirs(output_dir, exist_ok=True)

# Helper to load binary + meta
def load_dataset(path):
    meta_path = os.path.join(path, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    dtype = np.int64  # or int32 if that’s what you used
    train = np.fromfile(os.path.join(path, "train.bin"), dtype=dtype)
    val = np.fromfile(os.path.join(path, "val.bin"), dtype=dtype)

    return train, val, meta

train1, val1, meta1 = load_dataset(dataset1)
train2, val2, meta2 = load_dataset(dataset2)

# Merge
merged_train = np.concatenate([train1, train2])
merged_val = np.concatenate([val1, val2])

# Save
merged_train.tofile(os.path.join(output_dir, "train.bin"))
merged_val.tofile(os.path.join(output_dir, "val.bin"))

# Combine metas (we assume same tokenizer + vocab size)
merged_meta = {
    "train_len": int(len(merged_train)),
    "val_len": int(len(merged_val)),
    "vocab_size": meta1["vocab_size"],
    "tokenizer_path": meta1.get("tokenizer_path"),
    "eos_token": meta1.get("eos_token")
}

with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump(merged_meta, f)

print(f"🎉 Merged datasets saved to: {output_dir}")
