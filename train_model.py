from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("chatml_hf_tokenizer")

import numpy as np
import torch

def load_dataset(bin_path, block_size):
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    inputs = []
    labels = []
    for i in range(0, len(data) - block_size, block_size):
        x = torch.tensor(data[i:i+block_size], dtype=torch.long)
        y = torch.tensor(data[i+1:i+block_size+1], dtype=torch.long)
        inputs.append(x)
        labels.append(y)
    return list(zip(inputs, labels))

from transformers import LlamaConfig, LlamaForCausalLM

max_length = 512   # Maximum sequence length for the model
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    intermediate_size=2048,
    num_attention_heads=8,
    num_hidden_layers=6,
    max_position_embeddings=max_length,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

model = LlamaForCausalLM(config)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return {"input_ids": x, "labels": y}

train_data = MyDataset(load_dataset("./merged_dataset/train.bin", block_size=128))
val_data = MyDataset(load_dataset("./merged_dataset/val.bin", block_size=128))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    #evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    num_train_epochs=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=True,  # mixed precision if on supported GPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./model_output")