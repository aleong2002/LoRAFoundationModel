# main.py

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import RobertaTokenizer
from datasets import load_dataset

from preprocess import Preprocessor
from LoraLayer import LoRARobertaMLM
from model_eval import evaluate_lora_on_dart
from model_loader import save_checkpoint_to_drive, load_latest_checkpoint

# Set this to False to force retraining
USE_SAVED_CHECKPOINT = False

CHECKPOINT_DIR = "checkpoints"
FINAL_MODEL_PATH = "roberta_lora.pt"


def train(model, dataloader, optimizer, loss_fn, device,
          epochs=3, save_dir=CHECKPOINT_DIR):
    model.to(device)
    model.train()

    # Try to resume from latest checkpoint
    start_epoch = 0
    try:
        start_epoch = load_latest_checkpoint(model, optimizer, save_dir)
        print(f"Resuming training from epoch {start_epoch}")
    except Exception as e:
        print(f"No previous checkpoint found or failed to load ({e}). Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # CrossEntropy over vocab; ignore_index=-100 for non-mask positions
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
                print(f"[Epoch {epoch}] Step {step} - Loss: {loss.item():.4f}")

                # Optional GPU memory logging
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"  GPU memory allocated: {allocated:.2f} MB")
                    print(f"  GPU memory reserved: {reserved:.2f} MB")

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint at end of each epoch
        save_checkpoint_to_drive(model, optimizer, epoch, avg_loss, save_dir)

    return model


def prepare_datasets(preprocessor: Preprocessor):
   
    masked_dataset = load_dataset(
        "json",
        data_files={
            "train": "dataset/dart_masked_train.json",
            "test": "dataset/dart_masked_test.json"
        }
    )

    dataloader = preprocessor.get_dataloader(masked_dataset)
    return dataloader, masked_dataset


def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    preprocessor = Preprocessor(tokenizer, max_length=64, batch_size=8)

    dataloader, masked_dataset = prepare_datasets(preprocessor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LoRARobertaMLM()
    model.to(device)

    if not USE_SAVED_CHECKPOINT or not os.path.exists(FINAL_MODEL_PATH):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        print("Starting training...")
        model = train(model, dataloader, optimizer, loss_fn, device, epochs=3)

        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print(f"Final model saved to {FINAL_MODEL_PATH}")
    else:
        print(f"Skipping training. Using saved model at {FINAL_MODEL_PATH}")
        state_dict = torch.load(FINAL_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    print("\nStarting evaluation...")
    evaluate_lora_on_dart(
        model_path=FINAL_MODEL_PATH,
        masked_dataset=masked_dataset,
        tokenizer=tokenizer,
        device=device
    )


if __name__ == "__main__":
    main()
