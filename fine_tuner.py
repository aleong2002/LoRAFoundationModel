import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

from LoraLayer import LoRARobertaMLM
from model_eval import evaluate_lora_on_dart
from model_loader import save_checkpoint, load_latest_checkpoint


class LoRAFineTuner:
    def __init__(
        self,
        model_name="roberta-base",
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        max_length=64,
        batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        epochs=100,
        checkpoint_dir="checkpoints",
        output_dir="lora_outputs",
        save_every=5,
        eval_every=10
    ):

        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.save_every = save_every
        self.eval_every = eval_every
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training history
        self.history = {
            "train_loss": [],
            "eval_results": []
        }
    
    def load_datasets(self, train_file="dataset/dart_masked_train.json", 
                     test_file="dataset/dart_masked_test.json"):
        """Load DART masked datasets"""
        print("Loading DART masked datasets...")
        
        self.dataset = load_dataset(
            "json",
            data_files={
                "train": train_file,
                "test": test_file
            }
        )
        
        print(f"Train examples: {len(self.dataset['train'])}")
        print(f"Test examples: {len(self.dataset['test'])}")
        
        return self.dataset
    
    def create_dataloader(self, split="train"):
        """Create dataloader for the specified split"""
        
        def collate_fn(batch):
            """Process batch of examples"""
            input_texts = []
            target_texts = []
            
            for example in batch:
                input_text = example.get("input", "")
                target_text = example.get("target", "")
                
                # Replace [MASK] with tokenizer's mask token
                input_text = input_text.replace("[MASK]", self.tokenizer.mask_token)
                
                input_texts.append(input_text)
                target_texts.append(target_text)
            
            # Tokenize inputs
            encodings = self.tokenizer(
                input_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            # Create labels
            labels = input_ids.clone()
            
            # For each example, find mask position and set label
            for i, (input_id, target) in enumerate(zip(input_ids, target_texts)):
                # Initialize all positions to -100 (ignore)
                labels[i] = -100
                
                # Find mask token positions
                mask_positions = (input_id == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                
                if len(mask_positions) > 0:
                    # Tokenize target to get the token ID
                    target_encoding = self.tokenizer(
                        target,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )
                    
                    if target_encoding["input_ids"].size(1) > 0:
                        target_token_id = target_encoding["input_ids"][0, 0]
                        # Set label for first mask position
                        labels[i, mask_positions[0]] = target_token_id
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        dataloader = DataLoader(
            self.dataset[split],
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return dataloader
    
    def initialize_model(self):
        """Initialize LoRA model"""
        print("\nInitializing LoRA model...")
        self.model = LoRARobertaMLM(
            base_model_name=self.model_name,
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Initialize optimizer
        trainable_params_list = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params_list,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return self.model
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / max(1, num_batches)
        return avg_loss
    
    def train(self, resume_from_checkpoint=True, pretrained_model_path="roberta_lora.pt"):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING FINE-TUNING")
        print("=" * 60)
        
        # Load datasets
        self.load_datasets()
        train_dataloader = self.create_dataloader("train")
        
        # Initialize model
        self.initialize_model()
        
        # Try to load from pretrained model first
        start_epoch = 0
        if os.path.exists(pretrained_model_path):
            try:
                print(f"Loading pretrained model from {pretrained_model_path}...")
                state_dict = torch.load(pretrained_model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                print(f"Successfully loaded pretrained model!")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
        
        # Try to resume from checkpoint
        if resume_from_checkpoint:
            try:
                start_epoch = load_latest_checkpoint(
                    self.model,
                    self.optimizer,
                    self.checkpoint_dir
                )
                print(f"Resumed from checkpoint at epoch {start_epoch}")
            except Exception as e:
                print(f"No checkpoint found or failed to load: {e}")
                if not os.path.exists(pretrained_model_path):
                    print("Starting from scratch...")
        
        # Training loop
        print(f"\nTraining for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Log results
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Average loss: {avg_loss:.4f}")
            
            self.history["train_loss"].append({
                "epoch": epoch,
                "loss": avg_loss,
                "time": epoch_time
            })
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                print(f"Saving checkpoint at epoch {epoch}...")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    avg_loss,
                    self.checkpoint_dir
                )
            
            # Evaluate
            if (epoch + 1) % self.eval_every == 0:
                print(f"\nEvaluating at epoch {epoch}...")
                eval_results = self.evaluate()
                self.history["eval_results"].append({
                    "epoch": epoch,
                    **eval_results
                })
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "roberta_lora_final.pt")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save training history
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        return self.model
    
    def evaluate(self):
        """Evaluate the model on test set"""
        self.model.eval()
        
        # Save temporary model for evaluation
        temp_model_path = os.path.join(self.checkpoint_dir, "temp_eval_model.pt")
        torch.save(self.model.state_dict(), temp_model_path)
        
        # Run evaluation
        results = evaluate_lora_on_dart(
            model_path=temp_model_path,
            masked_dataset=self.dataset,
            tokenizer=self.tokenizer,
            device=self.device,
            output_dir=self.output_dir
        )
        
        return results


def main():
    
    config = {
        "model_name": "roberta-base",
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "max_length": 64,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "epochs": 10,
        "checkpoint_dir": "checkpoints",
        "output_dir": "lora_outputs",
        "save_every": 5,
        "eval_every": 10
    }
    
    # Initialize fine-tuner
    finetuner = LoRAFineTuner(**config)
    
    # Train the model (will automatically load from roberta_lora.pt if it exists)
    model = finetuner.train(
        resume_from_checkpoint=True,
        pretrained_model_path="roberta_lora.pt"
    )
    
    finetuner.evaluate()
    
    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()