import torch
import os
import time
import gc
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import re

from LoraLayer import LoRARobertaMLM


def load_model_from_checkpoint(model_path, device):
    """Load LoRA model from checkpoint file."""
    model = LoRARobertaMLM()
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    
    print(f"Model loaded from {model_path}")
    
    # Verify model state
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model - Trainable: {trainable_params:,} / Total: {total_params:,}")
    
    return model


def evaluate_lora_on_dart(model_path, masked_dataset, tokenizer, device, output_dir="lora_outputs"):
    """
    Evaluate LoRA model on DART masked language modeling task.
    
    Args:
        model_path: Path to saved model checkpoint (.pt file)
        masked_dataset: Dataset with examples containing 'input' and 'target'
        tokenizer: RoBERTa tokenizer
        device: torch device (cuda/cpu)
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model from checkpoint
    model = load_model_from_checkpoint(model_path, device)

    # Evaluation setup
    rouge = Rouge()
    results = []
    start_time = time.time()
    max_memory = 0
    correct = 0
    total = 0
    skipped = 0

    test_data = masked_dataset["test"]
    print(f"\nEvaluating on {len(test_data)} test examples...")

    # Evaluate each example
    for idx, example in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            input_text = example['input']
            target_text = example['target']
            
            # Replace [MASK] with tokenizer's mask token if present
            if '[MASK]' in input_text:
                masked_text = input_text.replace('[MASK]', tokenizer.mask_token)
            else:
                # Skip examples without mask for now
                skipped += 1
                continue
            
            # Tokenize
            encoding = tokenizer(
                masked_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Find [MASK] position
            mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            
            if len(mask_positions[1]) == 0:
                skipped += 1
                continue
                
            mask_index = mask_positions[1][0].item()
            
            # Get top-10 predictions for analysis
            top_k = torch.topk(logits[0, mask_index], k=10)
            top_predictions = [tokenizer.decode([idx.item()]).strip() for idx in top_k.indices]
            top_scores = [score.item() for score in top_k.values]
            
            # Get top prediction
            predicted_token = top_predictions[0]
            
            # Get target (first word/token of target for comparison)
            target_tokens = target_text.split()
            target_token = target_tokens[0] if target_tokens else target_text
            
            # Clean for comparison
            predicted_clean = predicted_token.lower().strip()
            target_clean = target_token.lower().strip()
            
            # Check correctness
            is_correct = predicted_clean == target_clean
            correct += int(is_correct)
            total += 1

            # Calculate BLEU and ROUGE scores
            try:
                pred_words = predicted_token.split() if predicted_token else ["unk"]
                target_words = target_token.split() if target_token else ["unk"]
                bleu = sentence_bleu([target_words], pred_words)
            except:
                bleu = 0.0
                
            try:
                rouge_score = rouge.get_scores(
                    predicted_token if predicted_token else "unk", 
                    target_token if target_token else "unk"
                )[0]["rouge-l"]["f"]
            except:
                rouge_score = 0.0

            # Save result
            result = {
                "Example_ID": idx,
                "Original Input": input_text,
                "Masked Input": masked_text,
                "Target Full": target_text,
                "Target Token": target_token,
                "Prediction": predicted_token,
                "Correct": is_correct,
                "BLEU": round(bleu, 3),
                "ROUGE-L": round(rouge_score, 3)
            }
            
            # Add top predictions
            for i in range(min(5, len(top_predictions))):
                result[f"Top_{i+1}"] = top_predictions[i]
                result[f"Score_{i+1}"] = round(top_scores[i], 3)
            
            results.append(result)

            # GPU memory tracking
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
                max_memory = max(max_memory, mem)
                
            # Print some examples
            if idx < 10 or (idx % 1000 == 0):
                print(f"\nExample {idx}:")
                print(f"  Input: {input_text}")
                print(f"  Target: {target_token}")
                print(f"  Predicted: {predicted_token}")
                print(f"  Top-3: {top_predictions[:3]}")
                print(f"  Correct: {is_correct}")
                
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue

    end_time = time.time()

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nProcessing complete. Total: {total}, Skipped: {skipped}")

    # Save results
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{output_dir}/lora_masked_evaluation.csv", index=False)
        
        # Save sample results
        sample_size = min(50, len(results_df))
        results_df.head(sample_size).to_json(
            f"{output_dir}/lora_masked_samples.json", 
            orient="records", 
            indent=2
        )

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        avg_bleu = results_df["BLEU"].mean()
        avg_rouge = results_df["ROUGE-L"].mean()
    else:
        accuracy = 0
        avg_bleu = 0
        avg_rouge = 0
        print("WARNING: No results to save!")

    # Efficiency logging
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    checkpoint_size = os.path.getsize(model_path) / 1e6

    eff_df = pd.DataFrame({
        "Model": ["RoBERTa + LoRA (Masked MLM)"],
        "Total Params": [f"{total_params:,}"],
        "Trainable Params": [f"{trainable_params:,}"],
        "Trainable %": [f"{100 * trainable_params / total_params:.2f}%"],
        "Evaluation Time (s)": [round(end_time - start_time, 2)],
        "Max GPU Memory (GB)": [round(max_memory, 2)],
        "Checkpoint Size (MB)": [round(checkpoint_size, 2)],
        "Total Examples": [len(test_data)],
        "Evaluated": [total],
        "Skipped": [skipped],
        "Accuracy": [f"{correct}/{total} ({accuracy:.2%})"],
        "Avg BLEU": [round(avg_bleu, 3)],
        "Avg ROUGE-L": [round(avg_rouge, 3)]
    })
    eff_df.to_csv(f"{output_dir}/lora_efficiency.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total examples: {len(test_data)}")
    print(f"Evaluated: {total}")
    print(f"Skipped: {skipped}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.2%})")
    print(f"Average BLEU: {avg_bleu:.3f}")
    print(f"Average ROUGE-L: {avg_rouge:.3f}")
    print(f"Evaluation time: {end_time - start_time:.2f}s")
    print(f"Max GPU memory: {max_memory:.2f} GB")
    print(f"Results saved to: {output_dir}")
    print("="*60)

    return {
        "accuracy": accuracy,
        "bleu": avg_bleu,
        "rouge": avg_rouge,
        "correct": correct,
        "total": total,
        "skipped": skipped
    }