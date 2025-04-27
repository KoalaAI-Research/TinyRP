import os
from datasets import load_dataset
import json

# Load your dataset
dataset_name = "roleplay4fun/aesir-v1.1"  # e.g. "HuggingFaceH4/ultrachat_200k"
split = "train"  # or "test", "validation" if needed
dataset = load_dataset(dataset_name, split=split)

# Output folder
output_dir = "chatml_formatted_dataset"
os.makedirs(output_dir, exist_ok=True)

# Role mapping for ChatML tags
role_map = {
    "system": "<|system|>",
    "human": "<|user|>",
    "gpt": "<|assistant|>",
}

import re

def clean_value(text):
    lines = text.strip().splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Remove line content like "{{user}}:" or any template-y nonsense
        if re.match(r"\{\{.*\}\}:", line):
            line = line.split(":", 1)[1].strip()

        # Remove "Name: " or "Name: " from the start of lines
        if re.match(r"^[A-Z][a-zA-Z0-9_ ]+: ", line):
            line = line.split(": ", 1)[1]

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def convert_to_chatml(conversation):
    chatml_text = ""
    for turn in conversation:
        role = turn["from"]
        value = clean_value(turn["value"])
        tag = role_map.get(role, "<|unknown|>")
        if value:  # only add if there's actually something to say
            chatml_text += f"{tag}\n{value}\n"
    return chatml_text.strip()

# Iterate through dataset and save each example as a .txt file
for i, example in enumerate(dataset):
    conversation = example.get("conversations")
    if not conversation:
        continue
    chatml_formatted = convert_to_chatml(conversation)
    with open(os.path.join(output_dir, f"example_{i:05d}.txt"), "w", encoding="utf-8") as f:
        f.write(chatml_formatted)

print(f"✅ Done! Saved {len(dataset)} files to '{output_dir}'")
