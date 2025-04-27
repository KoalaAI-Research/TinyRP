import os
import json
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

# Settings
dataset_name = "roleplay4fun/pippa"
split = "train"  # or whatever
output_dir = "tokenized_chatml_bin"
tokenizer_name = "chatml_tokenizer"  # or your model's tokenizer
train_split_ratio = 0.9

os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("chatml_hf_tokenizer")

# ChatML role mapping
def role_to_tag(role):
    if role.lower() == "user":
        return "<|user|>"
    else:
        return "<|assistant|>"

# Convert to ChatML-style string
def convert_roleplay_convo(convo):
    result = ""
    for turn in convo:
        role_tag = role_to_tag(turn["role"])
        content = turn["content"].strip()
        result += f"{role_tag}\n{content}\n"
    return result.strip()

# Load dataset and process
dataset = load_dataset(dataset_name, split=split)
all_convos = []

for example in dataset:
    convo = example.get("conversation")
    system = example.get("memory")
    if convo:
        chatml_text = system + "\n" + convert_roleplay_convo(convo)
        all_convos.append(chatml_text)

# Shuffle and split
random.shuffle(all_convos)
split_idx = int(len(all_convos) * train_split_ratio)
train_texts = all_convos[:split_idx]
val_texts = all_convos[split_idx:]

# Tokenize and flatten
def tokenize_and_flatten(texts):
    all_tokens = []
    for text in texts:
        ids = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
        all_tokens.extend(ids)
    return torch.tensor(all_tokens, dtype=torch.long)

train_tokens = tokenize_and_flatten(train_texts)
val_tokens = tokenize_and_flatten(val_texts)

# Save as .bin files
train_tokens.numpy().tofile(os.path.join(output_dir, "train.bin"))
val_tokens.numpy().tofile(os.path.join(output_dir, "val.bin"))

# Save metadata for training
with open(os.path.join(output_dir, "meta.json"), "w") as f:
    json.dump({
        "train_len": len(train_tokens),
        "val_len": len(val_tokens),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer_name
    }, f)

print(f"✅ Saved tokenized train/val splits to '{output_dir}'")
