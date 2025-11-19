import os
import time
import torch
import torch.nn as nn
from transformers import RobertaTokenizer
from torch.optim import AdamW

from preprocess import Preprocessor
from datasets import load_dataset

from LoraLayer import LoRARobertaMLM
from model_eval import evaluate_lora_on_dart
from model_loader import save_checkpoint_to_drive, load_latest_checkpoint

USE_SAVED_CHECKPOINT = True


def train(model, dataloader, optimizer, loss_fn, device, epochs=3, save_dir="checkpoints"):
    """Train the model with checkpointing support."""
    model.train()
    model.to(device)

    start_epoch = load_latest_checkpoint(model, optimizer, save_dir)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

                # GPU memory logging
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU memory allocated: {allocated:.2f} MB")
                    print(f"GPU memory reserved: {reserved:.2f} MB")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint at end of each epoch
        save_checkpoint_to_drive(model, optimizer, epoch, avg_loss)

    return model


def prepare_datasets(preprocessor):
    """Load and prepare masked datasets."""
    masked_dataset = load_dataset("json", data_files={
        "train": "dataset/dart_masked_train.json",
        "test": "dataset/dart_masked_test.json"
    })
    return preprocessor.get_dataloader(masked_dataset), masked_dataset


def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    preprocessor = Preprocessor(tokenizer)
    dataloader, masked_dataset = prepare_datasets(preprocessor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LoRARobertaMLM()
    model.to(device)
    
    if not USE_SAVED_CHECKPOINT:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        print("Starting training...")
        model = train(model, dataloader, optimizer, loss_fn, device, epochs=3)
        
        final_model_path = "roberta_lora.pt"
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    print("\nStarting evaluation...")
    evaluate_lora_on_dart(
        model_path="roberta_lora.pt",
        masked_dataset=masked_dataset,
        tokenizer=tokenizer,
        device=device
    )

if __name__ == "__main__":
    main()