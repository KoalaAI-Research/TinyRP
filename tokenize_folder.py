import os
import random
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

# Config
input_dir = "chatml_formatted_dataset"  # your .txt folder
output_dir = "final_tokenized_bin"
tokenizer_name = "chatml_hf_tokenizer"  # swap with your model's tokenizer
train_split_ratio = 0.9

os.makedirs(output_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

# Read all .txt files
def load_text_files(folder):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                texts.append(f.read().strip())
    return texts

all_texts = load_text_files(input_dir)
random.shuffle(all_texts)
split_idx = int(len(all_texts) * train_split_ratio)
train_texts = all_texts[:split_idx]
val_texts = all_texts[split_idx:]

# Tokenize & flatten
def tokenize_and_flatten(texts):
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
        all_tokens.extend(tokens)
    return torch.tensor(all_tokens, dtype=torch.long)

train_tokens = tokenize_and_flatten(train_texts)
val_tokens = tokenize_and_flatten(val_texts)

# Save .bin files
train_tokens.numpy().tofile(os.path.join(output_dir, "train.bin"))
val_tokens.numpy().tofile(os.path.join(output_dir, "val.bin"))

# Metadata
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump({
        "train_len": len(train_tokens),
        "val_len": len(val_tokens),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer_name
    }, f)

print(f"🎉 Done! Tokenized data saved to '{output_dir}'")
