import os
import pandas as pd
import argparse
import re
import spacy

# Load spaCy for fallback glosado
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    nlp = None

def clean_prediction(text, task_name="MSLG2SPA"):
    """
    Higly aggressive cleaning:
    1. Handle nulls
    2. Remove all content between < and > (tags, metadata)
    3. Remove all newlines and tabs to ensure single-line output
    4. Remove common mBART special tokens as safety net
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Remove ANY content between brackets (like <Source: ...> or <tag>)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove control characters that would break the one-line-per-instance rule
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Force UPPERCASE for glosado tasks
    if task_name == "SPA2MSLG":
        text = text.upper()
        
    return text.strip()

def text_to_pseudo_gloss(text):
    """Fallback logic from glosado.py"""
    if nlp is None or not text:
        return str(text).upper()
    doc = nlp(text)
    gloss_tokens = []
    for token in doc:
        if token.is_punct or token.is_space: continue
        if token.pos_ in ['DET', 'ADP', 'CCONJ']: continue
        word = token.lemma_ if token.pos_ in ['VERB', 'AUX', 'NOUN', 'ADJ'] else token.text
        gloss_tokens.append(word.strip().upper())
    return " ".join(gloss_tokens)

def generate_submission(input_csv, output_txt, task_name="MSLG2SPA", test_file=None):
    """
    Reads inference results and outputs them in IberLEF strict format.
    Includes fallback to pseudo-glossing if prediction is empty for SPA2MSLG.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Could not find input predictions at {input_csv}")
        return
        
    df = pd.read_csv(input_csv)
    if "prediction" not in df.columns or "ID" not in df.columns:
        print("Error: Input CSV must contain 'ID' and 'prediction' columns.")
        return
        
    # Load original test set for fallback text
    test_lookup = {}
    if test_file and os.path.exists(test_file):
        test_df = pd.read_csv(test_file, sep="\t")
        text_col = "MSLG" if "MSLG2SPA" in task_name else "SPA"
        test_lookup = dict(zip(test_df["ID"], test_df[text_col]))

    ids = df["ID"].tolist()
    raw_preds = df["prediction"].tolist()
    
    with open(output_txt, 'w', encoding='utf-8', newline='\n') as f:
        for inst_id, pred in zip(ids, raw_preds):
            cleaned = clean_prediction(pred, task_name)
            
            # Application of fallback if empty
            if not cleaned and task_name == "SPA2MSLG" and inst_id in test_lookup:
                original_text = test_lookup[inst_id]
                cleaned = text_to_pseudo_gloss(original_text)

            # Ultra-last check to ensure no < or > remain
            cleaned = re.sub(r'[<>]', '', cleaned)
            
            f.write(f'"{inst_id}"\t"{cleaned}"\n')
                
    print(f"[{task_name}] Successfully formatted predictions into {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format MSLG-SPA predictions for submission")
    parser.add_argument("--input", type=str, required=True, help="Path to the CSV containing raw model predictions (ID and prediction columns)")
    parser.add_argument("--team", type=str, default="Antigravity", help="Team Name")
    parser.add_argument("--solution", type=str, default="Run1", help="Solution/Run Name")
    parser.add_argument("--task", type=str, choices=["MSLG2SPA", "SPA2MSLG"], default="MSLG2SPA", help="Subtask identifier")
    parser.add_argument("--test_file", type=str, help="Path to original test file for fallback text")
    
    args = parser.parse_args()
    
    output_filename = f"{args.team}_{args.solution}_{args.task}.txt"
    generate_submission(args.input, output_filename, args.task, args.test_file)
    
    # Example usage:
    # python submission_formatter.py --input test_results_A.csv --team MyTeam --solution BestModel --task MSLG2SPA
