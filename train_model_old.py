#!/usr/bin/env python3
# pretrain_chatml.py
import os
import torch
from transformers import (
    AutoTokenizer, LlamaConfig, LlamaForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset, Dataset

# --- Check for GPU availability and set device ---
def get_device_setup():
    if torch.cuda.is_available():
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda", True
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"Intel XPU detected")
        return "xpu", False  # Intel XPU might work better with fp32
    else:
        print("No GPU detected, using CPU")
        return "cpu", False

device, use_fp16 = get_device_setup()

# --- Load custom tokenizer with error handling ---
try:
    tokenizer = AutoTokenizer.from_pretrained("chatml_hf_tokenizer", use_fast=True)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"Error loading custom tokenizer: {e}")
    print("Falling back to default LLaMA tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# Ensure proper special tokens
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
print(f"Using pad_token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")
print(f"BOS token: {tokenizer.bos_token}, id: {tokenizer.bos_token_id}")
print(f"EOS token: {tokenizer.eos_token}, id: {tokenizer.eos_token_id}")

# --- Load text dataset from folder with error handling ---
def load_data_from_folder(data_dir="chatml_formatted_dataset"):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found!")
    
    data_files = {"train": []}
    found_files = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            data_files["train"].append(os.path.join(data_dir, filename))
            found_files += 1
    
    if found_files == 0:
        raise ValueError(f"No .txt files found in {data_dir}")
        
    print(f"Found {found_files} text files for training")
    return data_files

try:
    data_files = load_data_from_folder()
    dataset = load_dataset("text", data_files=data_files)
    print(f"Loaded dataset with {len(dataset['train'])} examples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Create a small dummy dataset for testing
    print("Creating a dummy dataset for testing")
    dummy_texts = [
        "This is a sample text for testing the ChatML model.",
        "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you!<|im_end|>"
    ] * 10
    dataset = Dataset.from_dict({"text": dummy_texts})
    dataset = dataset.train_test_split(test_size=0.2)

# --- Tokenize the text with proper sequence length ---
max_length = 512  # Match with config's max_position_embeddings

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=True
    )
    return outputs

# Process the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=["text"],
    desc="Tokenizing dataset"
)

def main():
    # --- Data collator with proper padding ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # we're doing causal LM
    )

    # --- Define the LLaMA model ---
    config = LlamaConfig(
        vocab_size=len(tokenizer),  # Use actual tokenizer vocabulary size
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=6,
        max_position_embeddings=max_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    print(f"Initializing model with config: {config}")
    model = LlamaForCausalLM(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Move model to device
    model = model.to(device)

    # --- Create evaluation dataset if available ---
    eval_dataset = tokenized_dataset["test"] if "test" in tokenized_dataset else None

    # --- Training Arguments with device-specific settings ---
    training_args = TrainingArguments(
        output_dir="./chatml-tiny-model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2 if eval_dataset else None,
        gradient_accumulation_steps=8,
        #evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        num_train_epochs=3,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-4,
        warmup_steps=100,
        fp16=use_fp16,  # Based on device detection
        report_to="none",
        # Add overflows tracking for XPU stability
        dataloader_num_workers=4,
        remove_unused_columns=True,
        # Add early stopping
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train the model ---
    print("Starting training...")
    trainer.train()

    # --- Save the model ---
    print("Saving model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":  
    main()