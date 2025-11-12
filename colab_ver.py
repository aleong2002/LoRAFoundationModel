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

drive_path = "/roberta_run1"
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

def save_checkpoint_to_drive(model, optimizer, epoch, loss, drive_path="/checkpoints"):
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
""" def preprocess(batch, tokenizer):
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
    return inputs """

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

def predict_masked(model, tokenizer, input_text):
    # Example: "Barack Obama was born in <mask>."
    masked_input = input_text.replace("[MASK]", tokenizer.mask_token)
    inputs = tokenizer(masked_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_index].argmax(dim=-1)
        return tokenizer.decode(predicted_token_id)

def evaluate_lora_on_dart(model_path, masked_dataset, tokenizer, output_dir="lora_outputs"):
    import torch, os, time, gc
    import pandas as pd
    from rouge import Rouge
    from nltk.translate.bleu_score import sentence_bleu

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = LoRARobertaMLM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # Evaluation setup
    rouge = Rouge()
    results = []
    start_time = time.time()
    max_memory = 0
    correct = 0
    total = 0

    # Evaluate each example
    for example in masked_dataset["test"]:
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        labels = example["labels"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Find [MASK] position
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        predicted_id = logits[0, mask_index].argmax().item()
        predicted_token = tokenizer.decode(predicted_id).strip()
        target_token = tokenizer.decode(labels[mask_index]).strip()

        is_correct = predicted_token == target_token
        correct += int(is_correct)
        total += 1

        # Optional: BLEU and ROUGE on single-token prediction
        bleu = sentence_bleu([[target_token.split()]], predicted_token.split())
        rouge_score = rouge.get_scores(predicted_token, target_token)[0]["rouge-l"]["f"]

        results.append({
            "Masked Input": tokenizer.decode(example["input_ids"], skip_special_tokens=True),
            "Target": target_token,
            "Prediction": predicted_token,
            "Correct": is_correct,
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

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/lora_masked_evaluation.csv", index=False)
    results_df.head(10).to_json(f"{output_dir}/lora_masked_samples.json", orient="records", indent=2)

    # Efficiency logging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_size = os.path.getsize(model_path) / 1e6

    eff_df = pd.DataFrame({
        "Model": ["RoBERTa + LoRA (Masked MLM)"],
        "Trainable Params": [f"{trainable_params:,}"],
        "Evaluation Time (s)": [round(end_time - start_time, 2)],
        "Max GPU Memory (GB)": [round(max_memory, 2)],
        "Checkpoint Size (MB)": [round(checkpoint_size, 2)],
        "Accuracy": [f"{correct}/{total} = {correct / total:.2%}"]
    })
    eff_df.to_csv(f"{output_dir}/lora_efficiency.csv", index=False)

    print(f"Evaluation complete. Accuracy: {correct}/{total} = {correct / total:.2%}")
    print(f"Results saved to: {output_dir}")
   
def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load DART dataset
    masked_dataset = load_dataset("json", data_files={
        "train": "/dataset/dart_masked_train.json",
        #"validation": "/dataset/dart_masked_dev.json",
        "test": "/dataset/dart_masked_test.json"
    })

    """ # Mask transformation
    def mask_transform(example):
        triples = example["tripleset"]
        ref = example["annotations"][0]["text"]
        if triples and len(triples[0]) == 3:
            subject, relation, obj = triples[0]
            masked_text = ref.replace(obj, tokenizer.mask_token)
            return {"input": masked_text, "target": obj}
        return {"input": ref, "target": ""}

    masked_dataset = dataset.map(mask_transform)
    """

    def preprocess(example):
        masked_input = example["input"].replace("[MASK]", tokenizer.mask_token)  # Just in case
        inputs = tokenizer(masked_input, padding="max_length", truncation=True, max_length=64)

        try:
            mask_index = inputs["input_ids"].index(tokenizer.mask_token_id)
        except ValueError:
            inputs["labels"] = [-100] * len(inputs["input_ids"])
            return inputs

        target_id = tokenizer.convert_tokens_to_ids(example["target"])
        labels = [-100] * len(inputs["input_ids"])
        labels[mask_index] = target_id
        inputs["labels"] = labels
        return inputs


    tokenized_train = masked_dataset["train"].map(preprocess, batched=False)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(tokenized_train, batch_size=2, shuffle=True)

    print(tokenized_train[0]["input_ids"])
    print(tokenizer.decode(tokenized_train[0]["input_ids"]))

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoRARobertaMLM()
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, loss_fn, device)

    torch.save(model.state_dict(), "roberta_lora.pt")
    #torch.save(model.state_dict(), "/content/drive/MyDrive/lora_experiments/roberta_run1/roberta_lora.pt")
    # model.load_state_dict(torch.load("roberta_lora.pt"))

    # Evaluate on masked test set
    tokenized_test = masked_dataset["test"].map(preprocess, batched=False)
    evaluate_lora_on_dart("roberta_lora.pt", tokenized_test, tokenizer)

if __name__ == "__main__":
    main()