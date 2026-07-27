"""
AraBERT fine-tuning pipeline for Arabic PII detection.
Follows akschneider1/arabic-pii-ner-model train_model.py / train_model_minimal.py methodology,
adapted for CPU-only training and integrated with our experiment tracker.

Usage:
    python -m arabic_pii.train                  # quick run (subset)
    python -m arabic_pii.train --full           # full Wojood + larger synthetic set
    python -m arabic_pii.train --epochs 3       # custom epoch count
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import mlflow

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from arabic_pii.preprocessing import NERPreprocessor
from arabic_pii.data_augmentation import DataAugmentation
from arabic_pii.training_monitor import (
    ExperimentTracker, PIIEvaluator, MLflowEpochCallback, setup_mlflow
)

MODEL_NAME = 'aubmindlab/bert-base-arabertv2'
OUTPUT_DIR = 'arabic_pii_model'


def build_datasets(args) -> tuple:
    """Run data augmentation pipeline (Step 1 from methodology) if needed."""
    train_csv = 'train_augmented.csv'
    val_csv = 'val_augmented.csv'
    test_csv = 'test_augmented.csv'

    aug = DataAugmentation()

    if not os.path.exists(train_csv) or args.rebuild_data:
        print("\n=== DATA AUGMENTATION ===")
        aug.build(
            max_wojood_sentences=args.max_train_sentences,
            synthetic_sentences=args.synthetic_sentences,
            output_path=train_csv,
        )
    else:
        print(f"Using existing {train_csv}")

    if not os.path.exists(val_csv) or args.rebuild_data:
        aug.build_val(max_sentences=args.max_val_sentences, output_path=val_csv)
    if not os.path.exists(test_csv) or args.rebuild_data:
        aug.build_test(max_sentences=args.max_test_sentences, output_path=test_csv)

    return train_csv, val_csv, test_csv


def preprocess(csv_path: str, preprocessor: NERPreprocessor, max_length: int, max_sentences: int = None) -> Dataset:
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} tokens from {csv_path}")
    examples = preprocessor.preprocess_dataset(df, max_length=max_length, max_sentences=max_sentences)
    return Dataset.from_dict({
        'input_ids': [ex.input_ids for ex in examples],
        'attention_mask': [ex.attention_mask for ex in examples],
        'labels': [ex.labels for ex in examples],
    })


def make_compute_metrics(evaluator: PIIEvaluator):
    def compute_metrics(eval_pred):
        return evaluator.compute(eval_pred.predictions, eval_pred.label_ids)
    return compute_metrics


def train(args):
    # ── MLflow setup ────────────────────────────────────────────────────────
    setup_mlflow(experiment_name='arabic-pii-ner')
    run_name = args.run_name or f'arabert_pii_{"full" if args.full else "quick"}'
    tracker = ExperimentTracker(run_name=run_name)

    # ── Step 1: Data pipeline ───────────────────────────────────────────────
    print("\n=== STEP 1: DATA PIPELINE ===")
    train_csv, val_csv, test_csv = build_datasets(args)

    # ── Step 2: Preprocessing ───────────────────────────────────────────────
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessor = NERPreprocessor(MODEL_NAME)

    print("Preprocessing train set...")
    train_dataset = preprocess(train_csv, preprocessor, args.max_seq_length)

    print("Preprocessing val set...")
    val_dataset = preprocess(val_csv, preprocessor, args.max_seq_length)

    print("Preprocessing test set...")
    test_dataset = preprocess(test_csv, preprocessor, args.max_seq_length)

    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # ── Step 3: Model initialization ────────────────────────────────────────
    print("\n=== STEP 3: MODEL INITIALIZATION ===")
    num_labels = len(preprocessor.label_to_id)
    print(f"Model: {MODEL_NAME}")
    print(f"Labels ({num_labels}): {list(preprocessor.label_to_id.keys())}")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=preprocessor.id_to_label,
        label2id=preprocessor.label_to_id,
    )

    evaluator = PIIEvaluator(preprocessor.id_to_label)

    # ── Step 4: Training arguments ──────────────────────────────────────────
    print("\n=== STEP 4: TRAINING ===")
    hparams = {
        'model': MODEL_NAME,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'epochs': args.epochs,
        'max_seq_length': args.max_seq_length,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'lr_scheduler': 'linear',
        'optimizer': 'adamw',
        'fp16': False,
        'train_sentences': len(train_dataset),
        'val_sentences': len(val_dataset),
        'test_sentences': len(test_dataset),
    }
    tracker.log_hyperparameters(hparams)
    print("Hyperparameters:", hparams)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        fp16=False,
        use_cpu=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to='none',
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(
        preprocessor.tokenizer, padding=True
    )
    mlflow_cb = MLflowEpochCallback(json_tracker=tracker)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(evaluator),
        callbacks=[mlflow_cb],
    )

    # ── MLflow run wraps everything from training → model artifact ──────────
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\nMLflow run ID: {run.info.run_id}")
        mlflow.log_params(hparams)

        print("\nStarting training (CPU)...")
        trainer.train()

        # ── Step 5: Evaluation ──────────────────────────────────────────────
        print("\n=== STEP 5: EVALUATION ===")
        print("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        final_metrics = {
            k.replace('eval_', ''): v
            for k, v in test_results.items()
            if k.startswith('eval_') and isinstance(v, (int, float))
        }

        print("\nTest results:")
        for k, v in final_metrics.items():
            print(f"  {k:25s} {v:.4f}")

        # Log final test metrics to MLflow
        mlflow.log_metrics({f'test_{k}': v for k, v in final_metrics.items()})

        # ── Step 6: Save model + log artifact ──────────────────────────────
        print(f"\n=== STEP 6: SAVING MODEL → {OUTPUT_DIR} ===")
        trainer.save_model(OUTPUT_DIR)
        preprocessor.tokenizer.save_pretrained(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, 'label_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'label_to_id': preprocessor.label_to_id,
                'id_to_label': preprocessor.id_to_label,
            }, f)

        mlflow.log_artifact(OUTPUT_DIR, artifact_path='model')
        mlflow.log_artifact(str(tracker.log_path), artifact_path='logs')

    tracker.finish(final_metrics)
    print(f"\nModel saved to: {OUTPUT_DIR}/")
    print(f"View all runs: mlflow ui --backend-store-uri mlruns")
    print("Training complete.")
    return trainer, final_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune AraBERT for Arabic PII NER')
    p.add_argument('--run-name', default=None, help='Experiment run name')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4, dest='batch_size')
    p.add_argument('--grad-accum', type=int, default=4, dest='grad_accum')
    p.add_argument('--learning-rate', type=float, default=3e-5, dest='learning_rate')
    p.add_argument('--max-seq-length', type=int, default=64, dest='max_seq_length')
    # Quick-run defaults: ~20-30 min on CPU. Use --full for the deliverable run (~3-4h).
    p.add_argument('--max-train-sentences', type=int, default=800, dest='max_train_sentences')
    p.add_argument('--max-val-sentences', type=int, default=200, dest='max_val_sentences')
    p.add_argument('--max-test-sentences', type=int, default=400, dest='max_test_sentences')
    p.add_argument('--synthetic-sentences', type=int, default=500, dest='synthetic_sentences')
    p.add_argument('--rebuild-data', action='store_true', dest='rebuild_data')
    p.add_argument('--full', action='store_true', help='Use full Wojood (15k train sentences, 5k synthetic)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.full:
        # Full deliverable run: ~3-4h on CPU
        args.max_train_sentences = 10000
        args.synthetic_sentences = 3000
        args.max_val_sentences = 1500
        args.max_test_sentences = 3000
        args.max_seq_length = 128
        args.epochs = 3
    train(args)
