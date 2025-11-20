import psutil
import torch
import time
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.optim import AdamW
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
        
def inject_lora(model, r=8, alpha=16, dropout=0.1):
    for i, layer in enumerate(model.roberta.encoder.layer):
        self_attn = layer.attention.self
        self_attn.query = LoRALinear(self_attn.query, r=r, alpha=alpha, dropout=dropout)
        self_attn.value = LoRALinear(self_attn.value, r=r, alpha=alpha, dropout=dropout)
    return model

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset = load_dataset("csv", data_files={
        "train": "./dataset/train.csv",
        "test": "./dataset/test.csv"
    })

def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

encoded = dataset.map(preprocess, batched=True)
encoded= encoded.rename_column("label-coarse", "labels")
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(encoded["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(encoded["test"], batch_size=32)


model_ft = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=6)
model_lora = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=6)
model_lora = inject_lora(model_lora)
model_lora.to(device)
model_ft.to(device)

# Freeze all parameters except LoRA and classifier
for name, param in model_lora.named_parameters():
    if "lora_" not in name and "classifier" not in name:
        param.requires_grad = False

def train_and_eval(model, name):
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    # Training
    start = time.time()
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        tqdm.write(f"{name} Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
    end = time.time()
    print(f"{name} Training Time: {end - start:.2f} seconds")
    torch.save(model.state_dict(), f"{name}_model.pt") # save model weights

    # Evaluation
    model.eval()
    preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            losses.append(outputs.loss.item())

    acc = accuracy_score(all_labels, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Average Cross-EntropyLoss: {sum(losses) / len(losses):.4f}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{name} Trainable Params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{name} Peak GPU Memory Usage: {peak_memory:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
    else:
        import psutil
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 ** 2)
        print(f"{name} Peak CPU Memory Usage: {cpu_mem:.2f} MB")

train_and_eval(model_ft, "Fine-Tuned RoBERTa")
train_and_eval(model_lora, "LoRA RoBERTa")