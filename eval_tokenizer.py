import matplotlib.pyplot as plt
from tokenizers import Tokenizer
import numpy as np

# Load your trained tokenizer
tokenizer = Tokenizer.from_file("chatml_tokenizer.json")

# Load some eval text samples (maybe 500–1k ChatML-formatted lines)
with open("chatml_eval.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Stats: token counts per line
token_counts = [len(tokenizer.encode(line).tokens) for line in lines]

# --- 1. Basic Stats ---
print("🔍 Tokenizer Evaluation")
print(f"Vocab size: {tokenizer.get_vocab_size()}")
print(f"Total samples: {len(lines)}")
print(f"Average tokens/sample: {np.mean(token_counts):.2f}")
print(f"Max tokens in a single sample: {max(token_counts)}")

# --- 2. Plot Histogram ---
plt.figure(figsize=(8, 5))
plt.hist(token_counts, bins=30, color="#6c5ce7", edgecolor="black")
plt.title("📊 Token Count per Sample")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("token_length_histogram.png")
print("📈 Histogram saved as token_length_histogram.png")

# --- 3. Tokenization Examples (Debug Print) ---
print("\n🧪 Sample Tokenizations:")
for line in lines[:5]:
    enc = tokenizer.encode(line)
    print(f"Input: {line}\nTokens: {enc.tokens}\n---")
