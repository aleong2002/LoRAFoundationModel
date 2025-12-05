import os
import time
import math
import argparse
import psutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from sklearn.metrics import accuracy_score

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from preprocess import Preprocessor
from LoraLayer import LoRARobertaMLM, inject_lora
from model_eval import evaluate_lora_on_dart
from model_loader import save_checkpoint_to_drive, load_latest_checkpoint

USE_SAVED_CHECKPOINT = False
CHECKPOINT_DIR = "checkpoints"
FINAL_MLM_MODEL_PATH = "roberta_lora_mlm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlm(model, dataloader, optimizer, loss_fn, device,
              epochs=3, save_dir=CHECKPOINT_DIR, final_model_path=FINAL_MLM_MODEL_PATH):
    model.to(device)
    model.train()

    start_epoch = 0
    try:
        start_epoch = load_latest_checkpoint(model, optimizer, save_dir)
        print(f"[DART] Resuming training from epoch {start_epoch}")
    except Exception as e:
        print(f"[DART] No previous checkpoint found or failed to load ({e}). Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if step % 100 == 0:
                print(f"[DART][Epoch {epoch}] Step {step} - Loss: {loss.item():.4f}")

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"  GPU memory allocated: {allocated:.2f} MB")
                    print(f"  GPU memory reserved: {reserved:.2f} MB")

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"[DART] Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        save_checkpoint_to_drive(model, optimizer, epoch, avg_loss, save_dir)

    torch.save(model.state_dict(), final_model_path)
    print(f"[DART] Final MLM model saved to {final_model_path}")
    return model


def prepare_dart_datasets(preprocessor: Preprocessor):
    masked_dataset = load_dataset(
        "json",
        data_files={
            "train": "dataset/dart_masked_train.json",
            "test": "dataset/dart_masked_test.json"
        }
    )
    dataloader = preprocessor.get_dataloader(masked_dataset)
    return dataloader, masked_dataset


def run_dart(args):
    print("[DART] Using device:", DEVICE)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    preprocessor = Preprocessor(tokenizer, max_length=64, batch_size=args.batch_size)

    dataloader, masked_dataset = prepare_dart_datasets(preprocessor)

    model = LoRARobertaMLM()
    model.to(DEVICE)

    if not USE_SAVED_CHECKPOINT or not os.path.exists(FINAL_MLM_MODEL_PATH):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        model = train_mlm(
            model,
            dataloader,
            optimizer,
            loss_fn,
            device=DEVICE,
            epochs=args.epochs,
            save_dir=CHECKPOINT_DIR,
            final_model_path=FINAL_MLM_MODEL_PATH
        )
    else:
        print(f"[DART] Skipping training. Using saved model at {FINAL_MLM_MODEL_PATH}")
        state_dict = torch.load(FINAL_MLM_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

    print("\n[DART] Starting evaluation on DART...")
    evaluate_lora_on_dart(
        model_path=FINAL_MLM_MODEL_PATH,
        masked_dataset=masked_dataset,
        tokenizer=tokenizer,
        device=DEVICE
    )


def prepare_trec_datasets(max_length=128, train_bs=16, test_bs=32):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    dataset = load_dataset("csv", data_files={
        "train": "./dataset/train.csv",
        "test": "./dataset/test.csv"
    })

    def preprocess(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    encoded = dataset.map(preprocess, batched=True)
    encoded = encoded.rename_column("label-coarse", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(encoded["train"], batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(encoded["test"], batch_size=test_bs)
    return train_loader, test_loader


def train_and_eval_trec(model, name, train_loader, test_loader, epochs=3, lr=2e-5):
    model.to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start = time.time()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tqdm.write(f"{name} Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    end = time.time()
    print(f"{name} Training Time: {end - start:.2f} seconds")

    torch.save(model.state_dict(), f"{name.replace(' ', '_')}_model.pt")
    print(f"{name} model weights saved to {name.replace(' ', '_')}_model.pt")

    model.eval()
    preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            batch_labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )
            losses.append(outputs.loss.item())

    acc = accuracy_score(all_labels, preds)
    print(f"{name} accuracy: {acc:.4f}")
    print(f"{name} average cross-entropy Loss: {sum(losses) / len(losses):.4f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{name} trainable Params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{name} peak GPU Memory Usage: {peak_memory:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
    else:
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 ** 2)
        print(f"{name} Peak CPU Memory Usage: {cpu_mem:.2f} MB")


def run_trec(args):
    print("[TREC] Using device:", DEVICE)
    train_loader, test_loader = prepare_trec_datasets(
        max_length=args.max_length,
        train_bs=args.train_batch_size,
        test_bs=args.test_batch_size,
    )

    model_ft = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=args.num_labels
    )

    model_lora = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=args.num_labels
    )
    model_lora = inject_lora(
        model_lora,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    for name, param in model_lora.named_parameters():
        if "lora_" not in name and "classifier" not in name:
            param.requires_grad = False

    print("[TREC] training full fine-tuned RoBERTa")
    train_and_eval_trec(
        model_ft,
        "Fine-Tuned RoBERTa",
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr
    )
    print("[TREC] Training LoRA RoBERTa")
    train_and_eval_trec(
        model_lora,
        "LoRA RoBERTa",
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr
    )

def main():
    parser = argparse.ArgumentParser(description="lora experiments")
    parser.add_argument(
        "--task",
        choices=["dart", "trec"],
        required=True,
        help="choose dataset, <dart> and <trec>"
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=16, help="TREC train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="TREC test batch size.")
    parser.add_argument("--num_labels", type=int, default=6, help="Number of labels for TREC classification.")

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank for TREC.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha for TREC.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout for TREC.")

    args = parser.parse_args()

    if args.task == "dart":
        run_dart(args)
    elif args.task == "trec":
        run_trec(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
