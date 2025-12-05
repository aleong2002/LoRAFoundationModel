import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import RobertaForSequenceClassification
import argparse
import json

from LoraLayer import *

# from paper and official lora code for references

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_lora_model(path, r, alpha=16, num_labels=6):
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
    model = inject_lora(model, r=r, alpha=alpha)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model


def extract_lora_weights(model):
    lora_data = []
    for i, layer in enumerate(model.roberta.encoder.layer):
        attn = layer.attention.self
        layer_data = {}
        for name, module in [('query', attn.query), ('value', attn.value)]:
            if hasattr(module, 'lora_A'):
                A = module.lora_A.weight.data.cpu().numpy()
                B = module.lora_B.weight.data.cpu().numpy()
                W_base = module.base.weight.data.cpu().numpy()
                scaling = module.scaling
                delta_W = (B @ A) * scaling
                layer_data[name] = {'A': A, 'B': B, 'delta_W': delta_W, 
                                    'W_base': W_base, 'scaling': scaling, 'r': module.r}
        lora_data.append(layer_data)
    return lora_data


def compute_subspace_similarity(U1, U2, i, j):
    U1_i = U1[:, :i]
    U2_j = U2[:, :j]
    return np.linalg.norm(U1_i.T @ U2_j, 'fro')**2 / min(i, j)

def plot_rank_comparison(lora_data_r8, lora_data_r64, layer_idx, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for row, name in enumerate(['query', 'value']):
        dW_r8 = lora_data_r8[layer_idx][name]['delta_W']
        dW_r64 = lora_data_r64[layer_idx][name]['delta_W']
        
        U_r8, _, _ = np.linalg.svd(dW_r8, full_matrices=False)
        U_r64, _, _ = np.linalg.svd(dW_r64, full_matrices=False)
        
        max_i = min(8, U_r8.shape[1])
        max_j = min(64, U_r64.shape[1])
        
        sim_matrix = np.zeros((max_i, max_j))
        for i in range(1, max_i + 1):
            for j in range(1, max_j + 1):
                sim_matrix[i-1, j-1] = compute_subspace_similarity(U_r8, U_r64, i, j)
        
        im = axes[row, 0].imshow(sim_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[row, 0].set_title(f'ΔW{name[0]} : r=8 vs r=64 (Layer {layer_idx})', fontsize=11)
        axes[row, 0].set_xlabel('j (top-j of r=64)')
        axes[row, 0].set_ylabel('i (top-i of r=8)')
        xticks = [0, 7, 15, 23, 31, 39, 47, 55, 63]
        xticks = [x for x in xticks if x < max_j]
        axes[row, 0].set_xticks(xticks)
        axes[row, 0].set_xticklabels([x+1 for x in xticks])
        axes[row, 0].set_yticks(range(max_i))
        axes[row, 0].set_yticklabels(range(1, max_i + 1))
        plt.colorbar(im, ax=axes[row, 0], label='φ(r=8, r=64, i, j)')
        
        sim_zoomed = sim_matrix[:, :8]
        im2 = axes[row, 1].imshow(sim_zoomed, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[row, 1].set_title(f'ΔW{name[0]} : Zoomed (Layer {layer_idx})', fontsize=11)
        axes[row, 1].set_xlabel('j (top-j of r=64)')
        axes[row, 1].set_ylabel('i (top-i of r=8)')
        axes[row, 1].set_xticks(range(8))
        axes[row, 1].set_xticklabels(range(1, 9))
        axes[row, 1].set_yticks(range(max_i))
        axes[row, 1].set_yticklabels(range(1, max_i + 1))
        plt.colorbar(im2, ax=axes[row, 1], label='φ(r=8, r=64, i, j)')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'rank_comparison_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_seed_comparison(lora_data_s1, lora_data_s2, layer_idx, seed1, seed2, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, name in enumerate(['query', 'value']):
        dW_s1 = lora_data_s1[layer_idx][name]['delta_W']
        dW_s2 = lora_data_s2[layer_idx][name]['delta_W']
        
        U_s1, _, _ = np.linalg.svd(dW_s1, full_matrices=False)
        U_s2, _, _ = np.linalg.svd(dW_s2, full_matrices=False)
        
        r = min(U_s1.shape[1], U_s2.shape[1], 8)
        
        sim_matrix = np.zeros((r, r))
        for i in range(1, r + 1):
            for j in range(1, r + 1):
                sim_matrix[i-1, j-1] = compute_subspace_similarity(U_s1, U_s2, i, j)
        
        im = axes[idx].imshow(sim_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=0.8)
        axes[idx].set_title(f'ΔW{name[0]} : seed {seed1} vs {seed2}', fontsize=11)
        axes[idx].set_xlabel(f'j (top-j of seed {seed2})')
        axes[idx].set_ylabel(f'i (top-i of seed {seed1})')
        axes[idx].set_xticks(range(r))
        axes[idx].set_xticklabels(range(1, r + 1))
        axes[idx].set_yticks(range(r))
        axes[idx].set_yticklabels(range(1, r + 1))
        plt.colorbar(im, ax=axes[idx], label='φ')
    
    d = lora_data_s1[0]['query']['delta_W'].shape[0]
    rand1, rand2 = np.random.randn(d, 8), np.random.randn(d, 8)
    U_r1, _, _ = np.linalg.svd(rand1, full_matrices=False)
    U_r2, _, _ = np.linalg.svd(rand2, full_matrices=False)
    
    sim_rand = np.zeros((8, 8))
    for i in range(1, 9):
        for j in range(1, 9):
            sim_rand[i-1, j-1] = compute_subspace_similarity(U_r1, U_r2, i, j)
    
    im = axes[2].imshow(sim_rand, cmap='viridis', aspect='auto', vmin=0, vmax=0.8)
    axes[2].set_title('Random Gaussian Baseline', fontsize=11)
    axes[2].set_xlabel('j')
    axes[2].set_ylabel('i')
    axes[2].set_xticks(range(8))
    axes[2].set_xticklabels(range(1, 9))
    axes[2].set_yticks(range(8))
    axes[2].set_yticklabels(range(1, 9))
    plt.colorbar(im, ax=axes[2], label='φ')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'seed_comparison_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='viz')
    parser.add_argument('--models_dir', type=str, default='lora_models')
    parser.add_argument('--output_dir', type=str, default='lora_visualizations')
    parser.add_argument('--num_labels', type=int, default=6)
    args = parser.parse_args()
    
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)
    models_dir = Path(args.models_dir)
    
    try:
        model_r8 = load_lora_model(models_dir / "LoRA_r8_seed42_model.pt", r=8, num_labels=args.num_labels)
        model_r64 = load_lora_model(models_dir / "LoRA_r64_seed42_model.pt", r=64, num_labels=args.num_labels)
        lora_r8 = extract_lora_weights(model_r8)
        lora_r64 = extract_lora_weights(model_r64)
        for layer in [0, 5, 11]:
            plot_rank_comparison(lora_r8, lora_r64, layer, save_dir)
        del model_r64
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        lora_r8 = None
    
    try:
        model_s42 = load_lora_model(models_dir / "LoRA_r8_seed42_model.pt", r=8, num_labels=args.num_labels)
        model_s123 = load_lora_model(models_dir / "LoRA_r8_seed123_model.pt", r=8, num_labels=args.num_labels)
        lora_s42 = extract_lora_weights(model_s42)
        lora_s123 = extract_lora_weights(model_s123)
        for layer in [0, 5, 11]:
            plot_seed_comparison(lora_s42, lora_s123, layer, 42, 123, save_dir)
        lora_data = lora_s42
        del model_s123
    except FileNotFoundError as e:
        print(f"Skipped: {e}")
        lora_data = lora_r8

if __name__ == "__main__":
    main()