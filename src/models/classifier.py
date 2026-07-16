from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from .training import HSDDataset, build_model_bundle, LABEL_NAMES
from ..utils.metrics import compute_classification_metrics, metrics_for_trainer
from ..utils.data import set_seed


class HSDClassifier:
    """Classifier cho Vietnamese Hate Speech Detection"""
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_labels: int = len(LABEL_NAMES),
        seed: int = 42,
    ):
        set_seed(seed)
        self.model_name = model_name
        self.num_labels = num_labels
        
        bundle = build_model_bundle(model_name)
        self.tokenizer = bundle.tokenizer
        self.model = bundle.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
    
    def train(
        self,
        train_df,
        dev_df,
        output_dir: str = "results/models",
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        max_length: int = 128,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
    ):
        
        # Tạo datasets
        train_dataset = HSDDataset(train_df, self.tokenizer, max_length)
        dev_dataset = HSDDataset(dev_df, self.tokenizer, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=metrics_for_trainer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{output_dir}/final_model")
        print(f"Model saved to {output_dir}/final_model")
        
        return trainer
    
    def predict(self, texts, max_length: int = 128):
        self.model.eval()
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions.cpu().numpy().tolist()
    
    def predict_proba(self, texts, max_length: int = 128):
        self.model.eval()
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        classifier = cls(model_name=path)
        classifier.tokenizer = tokenizer
        classifier.model = model
        classifier.model.to(classifier.device)
        
        return classifier