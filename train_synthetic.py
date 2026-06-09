import os
import torch
import pandas as pd
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

# Set up logging for console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def load_synthetic_data(data_path):
    """Loads synthetic pseudo-glossa dataset"""
    df = pd.read_csv(data_path, sep="|").dropna()
    
    # Split into 95% train, 5% dev
    train_df = df.sample(frac=0.95, random_state=42)
    dev_df = df.drop(train_df.index)
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(dev_df)

def preprocess_function(examples, tokenizer, max_length=128):
    # Add [LSM] token to pseudo-glossa
    inputs = ["[LSM] " + gloss for gloss in examples["orth"]]
    targets = examples["translation"]
    
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    # We load the weights that come OUT of Phase 1 (PHOENIX pre-training)
    # If PHOENIX pre-training wasn't fully run, you can fallback to "mBART-50-LSM" (just the token added)
    phase1_model_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-PHOENIX"
    fallback_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-LSM"
    
    synthetic_data_path = "f:/!Projects/MSLG-SPA 2026/synthetic_data/synthetic_mslg_es.csv"
    output_dir = "f:/!Projects/MSLG-SPA 2026/mBART-50-Synthetic"
    log_csv_path = os.path.join(output_dir, "synthetic_training_history.csv")
    
    logger.info("Starting Phase 2 Denoising Pre-training...")
    
    if os.path.exists(phase1_model_dir) and os.path.exists(os.path.join(phase1_model_dir, "config.json")):
        load_dir = phase1_model_dir
        logger.info(f"Loaded weights from Phase 1 PHOENIX training: {load_dir}")
    else:
        load_dir = fallback_dir
        logger.info(f"Phase 1 weights not found. Falling back to base initialized model: {load_dir}")

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Loading datasets...")
    train_dataset, dev_dataset = load_synthetic_data(synthetic_data_path)
    logger.info(f"Loaded {len(train_dataset)} train samples and {len(dev_dataset)} dev samples.")
    
    logger.info("Loading model and tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(load_dir, src_lang="es_XX", tgt_lang="es_XX")
    
    # Ensure MSL special tokens are recognized
    tokenizer.add_tokens(["dm-", "+", "#"], special_tokens=True)
    
    model = MBartForConditionalGeneration.from_pretrained(load_dir)
    model.resize_token_embeddings(len(tokenizer))
    
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
    
    # Optimization exactly as Phase 1 for RTX 3060 12GB
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=False,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
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
    
    logger.info(f"Starting Denoising Pre-training on Synthetic Data. Output will be saved to: {output_dir}")
    
    try:
        # Uncomment to run the full training loop
        # trainer.train()
        # logger.info(f"Training completed successfully. Saving final model to {output_dir}")
        # trainer.save_model(output_dir)
        # tokenizer.save_pretrained(output_dir)
        
        logger.info("Phase 2 training script generated successfully. (Training is commented out by default to avoid blocking the agent)")
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        raise e
