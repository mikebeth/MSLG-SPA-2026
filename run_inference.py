import os
import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
import argparse

def run_inference(model_path, test_path, output_csv, subtask="A", device="cuda"):
    """
    Runs inference on a test file and saves results to a CSV.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang="es_XX", tgt_lang="es_XX")
    
    # Ensure MSL special tokens are present (should be in the saved tokenizer from training)
    special_tokens = ["dm-", "+", "#"]
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    # Load test data (tab-separated with ID and MSLG/SPA)
    df = pd.read_csv(test_path, sep="\t")
    input_col = "MSLG" if subtask == "A" else "SPA"
    
    if input_col not in df.columns:
        print(f"Error: Column {input_col} not found in {test_path}")
        return

    results = []
    print(f"Starting inference for Subtask {subtask} ({len(df)} samples)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[input_col])
        inst_id = row["ID"]
        
        # Add [LSM] prefix if it's a gloss input
        if subtask == "A":
            input_text = f"[LSM] {text}"
        else:
            input_text = text
            
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                num_beams=5,
                max_length=128,
                early_stopping=True
            )
            
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        results.append({
            "ID": inst_id,
            "prediction": prediction.strip()
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on MSLG-SPA test sets")
    parser.add_argument("--model", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to MSLG2SPA_test.txt or SPA2MSLG_test.txt")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--subtask", type=str, choices=["A", "B"], required=True, help="A for MSLG2SPA, B for SPA2MSLG")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_inference(args.model, args.test_file, args.output, args.subtask, device)
