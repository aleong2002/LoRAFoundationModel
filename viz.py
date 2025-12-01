import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import RobertaForSequenceClassification
import argparse
import json

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

    def forward(self, x):
        return self.base(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling

def inject_lora(model, r=8, alpha=16, dropout=0.1):
    for layer in model.roberta.encoder.layer:
        attn = layer.attention.self
        attn.query = LoRALinear(attn.query, r=r, alpha=alpha, dropout=dropout)
        attn.value = LoRALinear(attn.value, r=r, alpha=alpha, dropout=dropout)
    return model

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
    
    plt.suptitle(f'Figure 3: Subspace Similarity r=8 vs r=64 (Layer {layer_idx})\n'
                 'Top singular directions of r=8 overlap significantly with r=64', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f'fig3_rank_comparison_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_rank_comparison_layer{layer_idx}.png")

def plot_seed_comparison(lora_data_s1, lora_data_s2, layer_idx, seed1, seed2, save_dir):
    """Figure 4: Compare subspaces learned with different random seeds"""
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
    
    plt.suptitle(f'Figure 4: Subspace Similarity Between Seeds (Layer {layer_idx})\n'
                 'ΔWq has higher intrinsic rank than ΔWv', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f'fig4_seed_comparison_layer{layer_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_seed_comparison_layer{layer_idx}.png")


def plot_rank_effect(results_path, save_dir):
    try:
        with open(results_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"  Results file not found: {results_path}")
        return
    
    ranks, accuracies, params = [], [], []
    for key, val in results.items():
        if 'seed42' in key and key != 'fine_tuned':
            r = int(key.split('_')[0][1:])
            ranks.append(r)
            accuracies.append(val['accuracy'])
            params.append(val['trainable_params'])
    
    if not ranks:
        print("  No rank data found in results")
        return
    
    sorted_idx = np.argsort(ranks)
    ranks = [ranks[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    params = [params[i] for i in sorted_idx]
    

    ft_acc = results.get('fine_tuned', {}).get('accuracy', None)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(ranks, accuracies, 'o-', linewidth=2, markersize=10, label='LoRA')
    if ft_acc:
        axes[0].axhline(y=ft_acc, color='r', linestyle='--', linewidth=2, label=f'Fine-tuned ({ft_acc:.3f})')
    axes[0].set_xlabel('Rank r', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title('Table 6: Effect of Rank on Performance', fontsize=12)
    axes[0].set_xscale('log', base=2)
    axes[0].legend()
    for r, acc in zip(ranks, accuracies):
        axes[0].annotate(f'{acc:.3f}', (r, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    axes[1].plot(ranks, [p/1e6 for p in params], 's-', linewidth=2, markersize=10, color='orange')
    axes[1].set_xlabel('Rank r', fontsize=12)
    axes[1].set_ylabel('Trainable Parameters (M)', fontsize=12)
    axes[1].set_title('Trainable Parameters vs Rank', fontsize=12)
    axes[1].set_xscale('log', base=2)
    
    plt.suptitle('Table 6 Style: A small rank often suffices!', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'table6_rank_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: table6_rank_effect.png")

def plot_w_deltaw_analysis(lora_data, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for row, name in enumerate(['query', 'value']):
        frob_dW_proj, frob_W_proj, frob_rand_proj = [], [], []
        frob_dW, amplification = [], []
        
        for layer_data in lora_data:
            W = layer_data[name]['W_base']
            dW = layer_data[name]['delta_W']
            r = layer_data[name]['r']
            
            U_dW, _, Vt_dW = np.linalg.svd(dW, full_matrices=False)
            U_W, _, Vt_W = np.linalg.svd(W, full_matrices=False)
            
            proj_dW = np.linalg.norm(U_dW[:, :r].T @ W @ Vt_dW[:r, :].T, 'fro')
            proj_W = np.linalg.norm(U_W[:, :r].T @ W @ Vt_W[:r, :].T, 'fro')
            
            rand_U = np.linalg.qr(np.random.randn(W.shape[0], r))[0]
            rand_V = np.linalg.qr(np.random.randn(W.shape[1], r))[0]
            proj_rand = np.linalg.norm(rand_U.T @ W @ rand_V, 'fro')
            
            frob_dW_proj.append(proj_dW)
            frob_W_proj.append(proj_W)
            frob_rand_proj.append(proj_rand)
            frob_dW.append(np.linalg.norm(dW, 'fro'))
            amplification.append(frob_dW[-1] / proj_dW if proj_dW > 1e-8 else 0)
        
        layers = range(len(lora_data))
        
        ax = axes[row, 0]
        x = np.arange(len(layers))
        width = 0.25
        ax.bar(x - width, frob_dW_proj, width, label='ΔW subspace', alpha=0.8)
        ax.bar(x, frob_W_proj, width, label='W top-r', alpha=0.8)
        ax.bar(x + width, frob_rand_proj, width, label='Random', alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('||U^T W V||_F')
        ax.set_title(f'{name.capitalize()}: W projection onto subspaces', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xticks(x[::2])
        
        ax = axes[row, 1]
        ax.bar(layers, amplification, alpha=0.8, color='steelblue')
        ax.axhline(y=np.mean(amplification), color='r', linestyle='--', label=f'Mean: {np.mean(amplification):.1f}x')
        ax.set_xlabel('Layer')
        ax.set_ylabel('||ΔW||_F / ||U^T W V||_F')
        ax.set_title(f'{name.capitalize()}: Amplification Factor', fontsize=11)
        ax.legend()
        
        ax = axes[row, 2]
        mid = len(lora_data) // 2
        W, dW = lora_data[mid][name]['W_base'], lora_data[mid][name]['delta_W']
        U_dW, _, _ = np.linalg.svd(dW, full_matrices=False)
        U_W, _, _ = np.linalg.svd(W, full_matrices=False)
        
        pcts = [1, 5, 10, 25, 50, 75, 100]
        sims = []
        r = lora_data[mid][name]['r']
        for pct in pcts:
            k = max(1, int(pct / 100 * min(W.shape)))
            sims.append(compute_subspace_similarity(U_dW[:, :r], U_W[:, :k], r, k))
        
        ax.plot(pcts, sims, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Top k% of W singular vectors')
        ax.set_ylabel('Similarity with ΔW')
        ax.set_title(f'{name.capitalize()}: ΔW vs W subspace (Layer {mid})', fontsize=11)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Table 7 / Section 7.3: W vs ΔW Correlation\nΔW amplifies directions NOT emphasized in W', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'table7_w_deltaw_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: table7_w_deltaw_analysis.png")

def plot_singular_value_spectrum(lora_data, save_dir):
    """Singular value spectrum of ΔW"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, name in enumerate(['query', 'value']):
        all_sv = []
        for ld in lora_data:
            _, S, _ = np.linalg.svd(ld[name]['delta_W'], full_matrices=False)
            S_norm = S / S[0] if S[0] > 0 else S
            all_sv.append(S_norm[:min(len(S), 20)])
        
        for i, sv in enumerate(all_sv):
            axes[idx].plot(range(1, len(sv)+1), sv, alpha=0.3 + 0.5*(i/len(all_sv)), linewidth=1)
        
        max_len = max(len(s) for s in all_sv)
        padded = [np.pad(s, (0, max_len-len(s)), constant_values=0) for s in all_sv]
        axes[idx].plot(range(1, max_len+1), np.mean(padded, axis=0), 'k-', linewidth=3, label='Mean')
        axes[idx].set_xlabel('Singular Value Index')
        axes[idx].set_ylabel('Normalized Value')
        axes[idx].set_title(f'ΔW{name[0]} Spectrum', fontsize=11)
        axes[idx].legend()
        axes[idx].set_yscale('log')
    
    plt.suptitle('Singular Value Spectrum: Rapid decay = low intrinsic rank', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'singular_value_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: singular_value_spectrum.png")

def plot_parameter_comparison(results_path, save_dir):
    try:
        with open(results_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        return
    
    ft = results.get('fine_tuned', {})
    lora_r8 = results.get('r8_seed42', {})
    
    if not ft or not lora_r8:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Fine-Tuned', 'LoRA (r=8)']
    params = [ft['trainable_params']/1e6, lora_r8['trainable_params']/1e6]
    accs = [ft['accuracy'], lora_r8['accuracy']]
    
    bars = ax.bar(methods, params, color=['#ff7f0e', '#2ca02c'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Trainable Parameters (Millions)', fontsize=12)
    ax.set_title('Parameter Efficiency: Fine-Tuning vs LoRA', fontsize=12)
    
    for bar, p, acc in zip(bars, params, accs):
        ax.annotate(f'{p:.2f}M\n(acc: {acc:.3f})', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    reduction = (1 - lora_r8['trainable_params'] / ft['trainable_params']) * 100
    ax.text(0.5, 0.95, f'Parameter Reduction: {reduction:.1f}%', transform=ax.transAxes, 
            ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: parameter_comparison.png")

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
    
    plot_rank_effect(models_dir / "training_results.json", save_dir)
   
    if lora_data:
        plot_w_deltaw_analysis(lora_data, save_dir)
    else:
        print("Skipped: No model data available")
    
    if lora_data:
        plot_singular_value_spectrum(lora_data, save_dir)
    plot_parameter_comparison(models_dir / "training_results.json", save_dir)


if __name__ == "__main__":
    main()