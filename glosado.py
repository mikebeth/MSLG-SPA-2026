import spacy
import pandas as pd
import random
from datasets import load_dataset
import os
import re

# Load the Spanish NLP model
print("Loading spaCy model...")
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Downloading es_core_news_sm...")
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

def extract_spanish_sentences(num_sentences=10000):
    """
    Downloads a portion of the multilingual TED Talks dataset or equivalent 
    to extract simple Spanish sentences.
    """
    print(f"Downloading/Extracting {num_sentences} Spanish sentences...")
    # Using 'opus_books' as a simple, high-quality Spanish source available on HuggingFace
    dataset = load_dataset("opus_books", "en-es", split="train")
    
    # Extract Spanish texts from the translation pairs
    spanish_sentences = [example['translation']['es'] for example in dataset]
    
    # Filter for reasonable length (not too short, not too long)
    filtered = [s for s in spanish_sentences if 15 <= len(s) <= 100]
    
    # Sample the required amount
    if len(filtered) > num_sentences:
        sampled = random.sample(filtered, num_sentences)
    else:
        sampled = filtered
        
    return sampled

def text_to_pseudo_gloss(text):
    """
    Transforms natural Spanish text into an approximation of MSL Gloss.
    Rules:
    1. Lemmatize verbs and nouns
    2. Remove stopwords (articles, prepositions, conjunctions)
    3. Remove punctuation
    4. Convert to UPPERCASE
    """
    doc = nlp(text)
    gloss_tokens = []
    
    for token in doc:
        # Skip punctuation, spaces, and stopwords (unless they are pronoun/crucial)
        if token.is_punct or token.is_space:
            continue
            
        # Spanish MSL often drops articles (el, la, un), prepositions (de, a, en) and conjunctions (y, o)
        if token.pos_ in ['DET', 'ADP', 'CCONJ']:
            continue
            
        # For verbs, we want the infinitive form (lemma)
        if token.pos_ == 'VERB' or token.pos_ == 'AUX':
            word = token.lemma_
        elif token.pos_ == 'PRON':
            # Keep pronouns (YO, TU, EL)
            word = token.text
        else:
            # For nouns, adjectives, etc, try to use lemma to remove plurals/gender where possible
            word = token.lemma_ if token.lemma_ else token.text
            
        # Clean up and uppercase
        word = word.strip().upper()
        if word:
            gloss_tokens.append(word)
            
    # Optional: Very basic Topic-Comment reordering (Move time/location to front, verbs to end)
    # This is a complex NLP task, we'll keep it simple for now and rely on dropping context
    
    return " ".join(gloss_tokens)

if __name__ == "__main__":
    output_dir = "f:/!Projects/MSLG-SPA 2026/synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get raw sentences
    raw_sentences = extract_spanish_sentences(10000)
    print(f"Extracted {len(raw_sentences)} sentences.")
    
    # 2. Process into glosses
    print("Converting to Pseudo-Glosas (this may take a minute)...")
    synthetic_pairs = []
    
    for i, text in enumerate(raw_sentences):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{len(raw_sentences)}...")
            
        gloss = text_to_pseudo_gloss(text)
        
        # Only keep if we actually generated a meaningful gloss
        if len(gloss.split()) >= 2:
            synthetic_pairs.append({
                "orth": gloss,
                "translation": text
            })
            
    # 3. Save to CSV
    df = pd.DataFrame(synthetic_pairs)
    output_path = os.path.join(output_dir, "synthetic_mslg_es.csv")
    df.to_csv(output_path, sep="|", index=False)
    
    print(f"\nSaved {len(df)} synthetic pairs to {output_path}")
    
    print("\nSample generation:")
    for _ in range(5):
        sample = df.sample(1).iloc[0]
        print(f"TEXT:  {sample['translation']}")
        print(f"GLOSS: {sample['orth']}\n")
