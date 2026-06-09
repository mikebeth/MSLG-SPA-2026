import pandas as pd
import json
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from evaluate import load

def load_phoenix_data(data_dir):
    """Loads PHOENIX-2014-T dataset"""
    train_df = pd.read_csv(os.path.join(data_dir, "PHOENIX-2014-T.train.corpus.csv"), sep="|")
    dev_df = pd.read_csv(os.path.join(data_dir, "PHOENIX-2014-T.dev.corpus.csv"), sep="|")
    test_df = pd.read_csv(os.path.join(data_dir, "PHOENIX-2014-T.test.corpus.csv"), sep="|")
    
    # Extract only the necessary columns: orth (gloss) and translation (text)
    train_data = train_df[['orth', 'translation']].dropna()
    dev_data = dev_df[['orth', 'translation']].dropna()
    test_data = test_df[['orth', 'translation']].dropna()
    
    return train_data, dev_data, test_data

def setup_model_and_tokenizer(model_dir):
    """Loads mBART model and tokenizer, adds [LSM] token."""
    print("Loading tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir, src_lang="de_DE", tgt_lang="de_DE")
    
    print("Adding [LSM] and GLOSS conventions as special tokens...")
    # Add special tokens for Sign Language Glosses. 
    # Even though PHOENIX is DGS (German Sign Language), 
    # we use [LSM] as requested for the whole project to keep it consistent.
    # We also add MSL specific prefixes/separators to avoid fragmentation.
    special_tokens_dict = {'additional_special_tokens': ['[LSM]', 'dm-', '+', '#']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    print("Loading model...")
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    
    # Resize model embeddings if new tokens were added
    if num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added_toks} tokens. New vocab size: {len(tokenizer)}")
        
    return model, tokenizer

if __name__ == "__main__":
    phoenix_dir = "f:/!Projects/MSLG-SPA 2026/PHOENIX-2014-T"
    mbart_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50"
    
    # 1. Load Data
    print("Loading PHOENIX data...")
    train, dev, test = load_phoenix_data(phoenix_dir)
    print(f"Train size: {len(train)}, Dev size: {len(dev)}, Test size: {len(test)}")
    
    print("\nSample Data:")
    print("Gloss:", train.iloc[0]['orth'])
    print("Text:", train.iloc[0]['translation'])
    
    # 2. Setup Model & Tokenizer
    model, tokenizer = setup_model_and_tokenizer(mbart_dir)
    
    # Save the updated tokenizer and model for training
    output_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-LSM"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Pre-training setup complete.")
