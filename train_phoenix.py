import os
import torch
import pandas as pd
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

def load_phoenix_data(data_dir):
    """Loads PHOENIX-2014-T dataset"""
    train_df = pd.read_csv(os.path.join(data_dir, "PHOENIX-2014-T.train.corpus.csv"), sep="|")
    dev_df = pd.read_csv(os.path.join(data_dir, "PHOENIX-2014-T.dev.corpus.csv"), sep="|")
    
    train_data = train_df[['orth', 'translation']].dropna()
    dev_data = dev_df[['orth', 'translation']].dropna()
    
    return Dataset.from_pandas(train_data), Dataset.from_pandas(dev_data)

def preprocess_function(examples, tokenizer, max_length=128):
    # Add [LSM] token prefix to inform the model this is Gloss input
    inputs = ["[LSM] " + gloss for gloss in examples["orth"]]
    targets = examples["translation"]
    
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

import logging
import json
import numpy as np

# Set up logging for console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    """Callback to print and save metrics correctly at each logging step or epoch"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["step"] = state.global_step
            self.history.append(logs)
            pd.DataFrame(self.history).to_csv(self.log_path, index=False)
            logger.info(f"Step {state.global_step}: {logs}")

if __name__ == "__main__":
    phoenix_dir = "f:/!Projects/MSLG-SPA 2026/PHOENIX-2014-T"
    mbart_lsm_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-LSM"
    output_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-PHOENIX"
    log_csv_path = os.path.join(output_dir, "training_history.csv")
    
    logger.info("Starting Phase 1 Pre-training script verification...")
    
    # Verify directories exist
    if not os.path.exists(mbart_lsm_dir):
        logger.error(f"Fatal: Base model directory missing: {mbart_lsm_dir}")
        exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Loading datasets...")
    train_dataset, dev_dataset = load_phoenix_data(phoenix_dir)
    logger.info(f"Loaded {len(train_dataset)} train samples and {len(dev_dataset)} dev samples.")
    
    logger.info("Loading model and tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(mbart_lsm_dir)
    
    # Ensure MSL special tokens are recognized across all phases
    tokenizer.add_tokens(["dm-", "+", "#"], special_tokens=True)
    
    model = MBartForConditionalGeneration.from_pretrained(mbart_lsm_dir)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set the languages for the tokenizer (German to German in Phoenix)
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"
    
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    tokenized_dev = dev_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        remove_columns=dev_dataset.column_names
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # Reducido para evitar Out of Memory en la RTX 3060 (12GB)
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Tamaño de lote efectivo = 2 * 4 = 8 (sin perder capacidad)
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=False,    # OMITIR la generacion lenta durante el pre-entrenamiento, solo validar la Loss (hace la validacion mucho mas rapida)
        fp16=True,                      # Precision Mixta (Super rapido en RTX Serie 3000)
        gradient_checkpointing=True,    # INTERCAMBIA calculo por memoria: Vital para modelos de 600M params en 12GB VRAM
        dataloader_num_workers=4,       # Aprovechar los nucleos del i7-12700KF para pasar datos mas rapido a la GPU
        load_best_model_at_end=True,    # Guardar siempre el modelo que rindio mejor en Loss
        push_to_hub=False,
        logging_dir='./logs',
        logging_steps=50,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback(log_csv_path)]
    )
    
    logger.info(f"Starting Pre-training on PHOENIX-14T. Output will be saved to: {output_dir}")
    logger.info(f"Training metrics will be continuously written to: {log_csv_path}")
    
    try:
        trainer.train()
        logger.info(f"Training completed successfully. Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Phase 1 verification passed and execution complete!")
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        raise e
