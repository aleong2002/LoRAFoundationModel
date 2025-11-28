# LoRA vs Full Fine-Tuning on RoBERTa

This repository contains experiments comparing **full fine-tuning** of RoBERTa with **Low-Rank Adaptation (LoRA)** across two tasks:
- **TREC Question Classification (NLU)**
- **DART Masked Language Modeling (MLM, supplemental)**

The goal is to evaluate the trade-offs between accuracy, efficiency, and resource usage when applying parameter-efficient fine-tuning methods.

---

## Key Contributions
- Implemented a reproducible pipeline for **full fine-tuning vs LoRA fine-tuning** using Hugging Face Transformers.
- Benchmarked performance on **TREC** (classification) and **DART** (masked language modeling).
- Measured **accuracy, loss, training/evaluation time, GPU memory usage, and parameter efficiency**.
- Highlighted LoRA’s ability to achieve **state-of-the-art accuracy on TREC** with <1% trainable parameters.
- Documented challenges in adapting RoBERTa to DART MLM and lessons learned.

---

## ⚙️ Experimental Setup
- **Model Backbone**: `roberta-base`
- **Fine-Tuning Variants**:
  - Full fine-tuning (all parameters trainable)
  - LoRA adapters (query/value projections + classifier head)
- **Optimizer**: AdamW (`lr=2e-5`)
- **Epochs**: 3
- **Batch Sizes**: Train = 16, Eval = 32
- **Metrics**:
  - TREC: Classification Accuracy, Cross-Entropy Loss
  - DART: Masked Token Accuracy, BLEU, ROUGE-L
  - Shared: Parameter Efficiency, Training/Eval Time, GPU Memory
---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/aleong2002/LoRAFoundationModel.git
   cd LoRAFoundationModel

2. Install dependencies:
   pip install torch transformers datasets scikit-learn psutil tqdm rouge nltk

3. Run experiments:
    - To train on TREC:
        python train_lora_roberta_terc.py

    - To train on DART:
        python main.py

    - Can also be run in CoLab for GPU access by cloning GitHub repo 
