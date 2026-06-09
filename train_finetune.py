import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import logging
from datasets import Dataset
from sklearn.model_selection import KFold
import evaluate

# Set up logging for console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load metrics based on subtask
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
try:
    comet = evaluate.load("comet")
except Exception as e:
    logger.warning("Unbabel-comet failed to load. Ensure the model is downloaded if using Subtask A.")
    comet = None

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["step"] = state.global_step
            self.history.append(logs)
            pd.DataFrame(self.history).to_csv(self.log_path, index=False)
            logger.info(f"Step {state.global_step}: {logs}")

def load_official_mslg(data_path):
    """Loads the official 489-pair MSLG-SPA training dataset"""
    if not os.path.exists(data_path):
        logger.error(f"Official dataset not found at {data_path}. Creating a dummy 10-pair dataset for testing.")
        # Create dummy data so the script doesn't completely fail while waiting for release
        dummy_data = {
            "orth": ["TÚ LLEGAR TARDE POR QUÉ", "AMÉRICA YO VIVIR"] * 5,
            "translation": ["¿Por qué llegaste tarde?", "Vivo en América."] * 5
        }
        return pd.DataFrame(dummy_data)
        
    # Load based on official txt format (tab separated, MSLG and SPA columns)
    df = pd.read_csv(data_path, sep="\t")
    if "MSLG" in df.columns and "SPA" in df.columns:
        df = df.rename(columns={"MSLG": "orth", "SPA": "translation"})
    return df

def preprocess_function(examples, tokenizer, max_length=128, subtask="A"):
    # Subtask A: Gloss (orth) -> Spanish (translation)
    # Subtask B: Spanish (translation) -> Gloss (orth)
    
    if subtask == "A":
        # MSLG2SPA
        inputs = ["[LSM] " + gloss for gloss in examples["orth"]]
        targets = examples["translation"]
        tokenizer.src_lang = "es_XX"  # mBART might not have 'LSM', we use es_XX as base with [LSM] prepended
        tokenizer.tgt_lang = "es_XX"
    else:
        # SPA2MSLG
        inputs = examples["translation"]
        targets = ["[LSM] " + gloss for gloss in examples["orth"]]
        tokenizer.src_lang = "es_XX"
        tokenizer.tgt_lang = "es_XX"
        
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer, subtask="A"):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = {}
    result["bleu"] = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    result["chrf"] = chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    result["meteor"] = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]
    
    # Subtask A heavily weights COMET as it's Spanish output
    if subtask == "A" and comet is not None:
        comet_score = comet.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels], sources=[""]*len(decoded_preds))
        result["comet"] = np.mean(comet_score["scores"])
        
    # Subtask B (SPA2MSLG) relies only on BLEU/chrF/METEOR
    
    # Calculate average
    result["mean_score"] = np.mean(list(result.values()))
    
    return {k: round(v, 4) for k, v in result.items()}

def train_kfold(df, base_model_dir, output_dir, subtask="A", k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        logger.info(f"=== Starting Fold {fold + 1}/{k} for Subtask {subtask} ===")
        
        fold_output_dir = os.path.join(output_dir, f"fold_{fold+1}")
        
        # Check if this fold has already been fully trained
        if os.path.exists(os.path.join(fold_output_dir, "model.safetensors")) or os.path.exists(os.path.join(fold_output_dir, "pytorch_model.bin")):
            logger.info(f"Fold {fold+1} already completed. Skipping...")
            continue
            
        os.makedirs(fold_output_dir, exist_ok=True)
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Load from base dir at the start of each fold to avoid data leakage
        tokenizer = MBart50TokenizerFast.from_pretrained(base_model_dir, src_lang="es_XX", tgt_lang="es_XX")
        
        # Add MSL specific special tokens to avoid fragmentation
        special_tokens = ["dm-", "+", "#"]
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        model = MBartForConditionalGeneration.from_pretrained(base_model_dir)
        # Resize embeddings to accommodate new tokens
        model.resize_token_embeddings(len(tokenizer))
        
        tokenized_train = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, subtask=subtask), 
            batched=True, remove_columns=train_dataset.column_names
        )
        tokenized_val = val_dataset.map(
            lambda x: preprocess_function(x, tokenizer, subtask=subtask), 
            batched=True, remove_columns=val_dataset.column_names
        )
        
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Hyperparameters specific to MSLG Fine-tuning (Very small dataset: 489 samples)
        # Learning rate is lower to avoid destroying pre-trained knowledge
        training_args = Seq2SeqTrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=10, # More epochs because dataset is tiny
            predict_with_generate=True, # Need this ON for fine-tuning to calculate BLEU/COMET
            generation_max_length=128,
            fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            load_best_model_at_end=True,
            metric_for_best_model="comet" if subtask == "A" else "bleu",
            push_to_hub=False,
            logging_dir=os.path.join(fold_output_dir, 'logs'),
            logging_steps=10,
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, subtask),
            callbacks=[LoggingCallback(os.path.join(fold_output_dir, "metrics.csv"))]
        )
        
        try:
            trainer.train()
            eval_results = trainer.evaluate()
            all_fold_results.append(eval_results)
            trainer.save_model(fold_output_dir)
            tokenizer.save_pretrained(fold_output_dir)
            logger.info(f"Fold {fold+1} training complete.")
        except Exception as e:
            logger.error(f"Fold {fold+1} failed: {e}")
            raise e
            
    return all_fold_results

if __name__ == "__main__":
    # Best model from Phase 2
    phase2_model_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-Synthetic"
    fallback_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-LSM"
    
    official_data_path = "f:/!Projects/MSLG-SPA 2026/MSLG_SPA_train.txt"
    output_base_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-MSLG-Finetuned"
    
    logger.info("Setting up Phase 3: Fine-Tuning Script...")
    
    if os.path.exists(phase2_model_dir) and os.path.exists(os.path.join(phase2_model_dir, "config.json")):
        base_model_dir = phase2_model_dir
        logger.info(f"Loaded Phase 2 Pre-trained Weights: {base_model_dir}")
    else:
        base_model_dir = fallback_dir
        logger.info(f"Phase 2 weights not found. Falling back to base model: {base_model_dir}")
        
    df = load_official_mslg(official_data_path)
    logger.info(f"Loaded dataset with {len(df)} samples.")
    
    # Setup for Subtask A (MSLG -> SPA)
    logger.info("Initializing Subtask A Cross-Validation...")
    train_kfold(df, base_model_dir, os.path.join(output_base_dir, "Subtask_A"), subtask="A", k=5)
    
    # Setup for Subtask B (SPA -> MSLG)
    logger.info("Initializing Subtask B Cross-Validation...")
    train_kfold(df, base_model_dir, os.path.join(output_base_dir, "Subtask_B"), subtask="B", k=5)
    
    logger.info("Phase 3 Script Validation Complete!")
