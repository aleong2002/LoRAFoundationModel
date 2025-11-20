import os
import time
import torch

from LoraLayer import LoRARobertaMLM

def load_model_from_checkpoint(model_path, device):
    """
    Load LoRA model from checkpoint (.pt).
    """
    model = LoRARobertaMLM()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    print(f"Model loaded from {model_path}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model - Trainable: {trainable_params:,} / Total: {total_params:,}")

    return model


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

def save_checkpoint_to_drive(model, optimizer, epoch, loss, drive_path="checkpoints"):
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
