# !pip install datasets transformers tqdm torch rouge

import os, time
import torch
import torch.nn as nn
import pandas as pd
import gc

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !git clone https://github.com/aleong2002/LoRAFoundationModel.git to access datasets

drive_path = "/content/drive/MyDrive/lora_experiments/roberta_run1"
def save_checkpoint(model, optimizer, epoch, loss, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, f"{save_dir}/checkpoint_{timestamp}.pt")

def load_latest_checkpoint(model, optimizer, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        return 0  # No checkpoint found

    checkpoints = sorted(os.listdir(save_dir), reverse=True)
    for ckpt in checkpoints:
        if ckpt.endswith(".pt"):
            path = os.path.join(save_dir, ckpt)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resumed from checkpoint: {ckpt}")
            return checkpoint["epoch"] + 1  # Resume from next epoch
    return 0

def save_checkpoint_to_drive(model, optimizer, epoch, loss, drive_path="/content/drive/MyDrive/checkpoints"):
    os.makedirs(drive_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"checkpoint_epoch{epoch}_{timestamp}.pt"
    filepath = os.path.join(drive_path, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, filepath)

    print(f"Checkpoint saved to: {filepath}")

# -------------------------------
# LoRA Layer Definition
# -------------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze base weights
        for param in self.parameters():
            param.requires_grad = False
        for param in self.lora_A.parameters():
            param.requires_grad = True
        for param in self.lora_B.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = x.to(self.lora_A.weight.device)
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
    
# -------------------------------
# LoRA-RoBERTa MLM Wrapper
# -------------------------------
class LoRARobertaMLM(nn.Module):
    def __init__(self, base_model_name="roberta-base", r=8, alpha=32):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained(base_model_name)
        self.inject_lora(r, alpha)

    def inject_lora(self, r, alpha):
      self.lora_modules = nn.ModuleList()  # Register all LoRA layers

      for name, module in self.model.named_modules():
          if isinstance(module, nn.Linear) and ("query" in name or "value" in name):
              in_dim = module.in_features
              out_dim = module.out_features
              lora = LoRALayer(in_dim, out_dim, r=r, alpha=alpha)
              self.lora_modules.append(lora)  # Register it so .to(device) works

              original_forward = module.forward
              module.forward = self._wrap_forward(original_forward, lora)
    def _wrap_forward(self, original_forward, lora_module):
        def wrapped(x):
            return original_forward(x) + lora_module(x)
        return wrapped

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess(batch, tokenizer):
    prompts = []
    targets = []

    for tripleset, annotations in zip(batch["tripleset"], batch["annotations"]):
        triples = [" ".join(triple) for triple in tripleset]
        prompt = " | ".join(triples) + " → <mask>"
        target = annotations[0]["text"] if annotations else ""

        prompts.append(prompt)
        targets.append(target)

    inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

def train(model, dataloader, optimizer, loss_fn, device, epochs=3, save_dir="checkpoints"):
    model.train()
    model.to(device)

    start_epoch = load_latest_checkpoint(model, optimizer, save_dir)
    last_save_time = time.time()

    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            print("Input IDs device:", input_ids.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save every 30 minutes
            """if time.time() - last_save_time > 1800:  # every 30 minutes
              save_checkpoint_to_drive(model, optimizer, epoch, loss.item())
              last_save_time = time.time()
            """

            print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

            # GPU memory logging
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"GPU memory allocated: {allocated:.2f} MB")
                print(f"GPU memory reserved: {reserved:.2f} MB")

        save_checkpoint_to_drive(model, optimizer, epoch, loss.item())

def evaluate_lora_on_dart(model_path, dataset, output_dir="lora_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare test set
    dart_test = dataset["test"]

    # Evaluation setup
    rouge = Rouge()
    results = []
    start_time = time.time()
    max_memory = 0

    # Generate predictions and compute metrics
    for example in dart_test:
        input_text = example["triples_as_text"]
        reference = example["target"]

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        pred_lora = tokenizer.decode(outputs[0], skip_special_tokens=True)

        bleu = sentence_bleu([reference.split()], pred_lora.split())
        rouge_score = rouge.get_scores(pred_lora, reference)[0]["rouge-l"]["f"]

        results.append({
            "Input": input_text,
            "Reference": reference,
            "LoRA Prediction": pred_lora,
            "BLEU": round(bleu, 3),
            "ROUGE-L": round(rouge_score, 3)
        })

        # GPU memory tracking
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            max_memory = max(max_memory, mem)
            torch.cuda.empty_cache()
            gc.collect()

    end_time = time.time()

    # Save full results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/lora_evaluation.csv", index=False)
    results_df.head(10).to_json(f"{output_dir}/lora_qualitative_samples.json", orient="records", indent=2)

    # Efficiency logging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_size = os.path.getsize(model_path) / 1e6

    eff_df = pd.DataFrame({
        "Model": ["RoBERTa + LoRA"],
        "Trainable Params": [f"{trainable_params:,}"],
        "Evaluation Time (s)": [round(end_time - start_time, 2)],
        "Max GPU Memory (GB)": [round(max_memory, 2)],
        "Checkpoint Size (MB)": [round(checkpoint_size, 2)]
    })
    eff_df.to_csv(f"{output_dir}/lora_efficiency.csv", index=False)

    print(f"✅ Evaluation complete. Results saved to: {output_dir}")
   
def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")    
    dataset = load_dataset("json", data_files={
        "train": "./dataset/dart-v1.1.1-full-train.json",
        "validation": "./dataset/dart-v1.1.1-full-dev.json",
        "test": "./dataset/dart-v1.1.1-full-test.json"
    })
    
    train_data = dataset["train"]
    #print(train_data[0])
    
    tokenized_train = dataset["train"].map(lambda x: preprocess(x, tokenizer), batched=True)    
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(tokenized_train, batch_size=2, shuffle=True)
    print(tokenized_train[0]["input_ids"])
    print(tokenizer.decode(tokenized_train[0]["input_ids"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoRARobertaMLM()
    model.to(device)
    """for name, param in model.named_parameters():
      if param.requires_grad:
          print(f"{name} → {param.device}")
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, dataloader, optimizer, loss_fn, device)
    torch.save(model.state_dict(), "roberta_lora.pt")
    torch.save(model.state_dict(), "/content/drive/MyDrive/checkpoints/roberta_lora.pt")
    evaluate_lora_on_dart("roberta_lora.pt", dataset)

if __name__ == "__main__":
    main()