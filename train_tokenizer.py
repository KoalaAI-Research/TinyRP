from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

import os
from glob import glob

# Point to your dataset folder
data_dir = "chatml_formatted_dataset"  # or wherever your .txt files live

# Grab all .txt files in the folder
corpus_files = glob(os.path.join(data_dir, "*.txt"))

print(f"📂 Found {len(corpus_files)} training files.")

# Define ChatML special tokens
special_tokens = [
    "<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant",
    "<|im_end|>", "<pad>", "<unk>"
]

# Build the tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.BPEDecoder()

# Post-processing to add BOS/EOS tokens (optional, for training convenience)
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A <|im_end|>",
    special_tokens=[
        ("<|im_end|>", 1),
    ]
)

# Train it
trainer = BpeTrainer(
    vocab_size=4096,  # can tweak based on dataset size
    special_tokens=special_tokens,
    show_progress=True,
)

tokenizer.train(files=corpus_files, trainer=trainer)

# Save
tokenizer.save("chatml_tokenizer.json")
print("✅ ChatML tokenizer saved.")

from transformers import PreTrainedTokenizerFast

# Convert the tokenizers.Tokenizer to a transformers tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    #cls_token="<s>",
    sep_token="<|im_end|>",
    mask_token="<mask>",  # Add if you plan to use for masked language modeling
    additional_special_tokens=[
        "<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"
    ]
)

# Save the tokenizer to a directory
tokenizer_save_path = "chatml_hf_tokenizer"
hf_tokenizer.save_pretrained(tokenizer_save_path)
print(f"✅ Hugging Face compatible tokenizer saved to {tokenizer_save_path}")
