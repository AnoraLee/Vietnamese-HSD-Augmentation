import sys
import os
sys.path.append('.')

import pandas as pd
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from src.utils.data import load_hf_dataset, save_data, load_config
from src.utils.metrics import LABEL_NAMES, metrics_for_trainer
from src.models.training import HSDDataset, build_model_bundle


def main():
    config = load_config()
    train_config = config['training']
    hf_config = config['huggingface']
    
    REPO_ID = hf_config['dataset_repo_id']
    MODEL_NAME = train_config['model_name']
    OUTPUT_DIR = train_config['output_dir']
    BATCH_SIZE = train_config['batch_size']
    LEARNING_RATE = train_config['learning_rate']
    EPOCHS = train_config['epochs']
    MAX_LENGTH = train_config['max_length']
    
    print(f"Loading data from {REPO_ID}...")
    
    try:
        train_df = load_hf_dataset(REPO_ID, split='tdtu_train')
    except:
        train_df = pd.read_csv('data/augmented/train_bt.csv')
    
    dev_df = pd.read_csv('data/processed/dev.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f"   Train: {len(train_df):,}")
    print(f"   Dev: {len(dev_df):,}")
    print(f"   Test: {len(test_df):,}")
    
    print("\nTrain label distribution:")
    print(train_df['label'].value_counts())
    
    print(f"\nBuilding model: {MODEL_NAME}")
    bundle = build_model_bundle(MODEL_NAME)
    tokenizer = bundle.tokenizer
    model = bundle.model
    
    print("\nCreating datasets...")
    train_dataset = HSDDataset(train_df, tokenizer, MAX_LENGTH)
    dev_dataset = HSDDataset(dev_df, tokenizer, MAX_LENGTH)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_ratio=train_config.get('warmup_ratio', 0.06),
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=metrics_for_trainer,
    )
    
    trainer.train()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
    print("\nPushing model to HuggingFace...")
    model.push_to_hub(f"{hf_config['model_repo_prefix']}_phobert", use_auth_token=True)
    tokenizer.push_to_hub(f"{hf_config['model_repo_prefix']}_phobert", use_auth_token=True)
    print(f"Model uploaded to HuggingFace!")
    
    test_dataset = HSDDataset(test_df, tokenizer, MAX_LENGTH)
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:")
    for key, value in test_results.items():
        print(f"   {key}: {value:.4f}")
    
    save_data(pd.DataFrame([test_results]), 'results/metrics/test_results.csv')
    
if __name__ == "__main__":
    main()