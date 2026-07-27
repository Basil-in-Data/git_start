"""
Training monitoring and experiment tracking.
Follows akschneider1/arabic-pii-ner-model training_monitor.py methodology.

Tracking backends:
  1. MLflow  — primary: hyperparameters, per-epoch metrics, model artifact, UI
  2. JSON    — fallback: lightweight local log, always written regardless of MLflow
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


PII_ENTITIES = ['PERSON', 'LOCATION', 'ORGANIZATION', 'PHONE', 'EMAIL', 'ID_NUMBER', 'ADDRESS']


# ── JSON tracker (always-on fallback) ────────────────────────────────────────

class ExperimentTracker:
    """Writes a single JSON log file per run — no external dependencies."""

    def __init__(self, log_dir: str = 'experiment_logs', run_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.run_name = run_name or f'run_{int(time.time())}'
        self.log_path = self.log_dir / f'{self.run_name}.json'
        self._record: Dict = {
            'run_name': self.run_name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': {},
            'epochs': [],
            'final_metrics': None,
        }
        self._flush()

    def log_hyperparameters(self, params: Dict):
        self._record['hyperparameters'] = params
        self._flush()

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        self._record['epochs'].append({'epoch': epoch, **{k: round(v, 4) for k, v in metrics.items()}})
        self._flush()

    def finish(self, final_metrics: Optional[Dict] = None):
        self._record['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if final_metrics:
            self._record['final_metrics'] = {k: round(v, 4) for k, v in final_metrics.items()}
        self._flush()
        print(f"JSON log saved → {self.log_path}")

    def _flush(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self._record, f, ensure_ascii=False, indent=2)


# ── MLflow helpers ────────────────────────────────────────────────────────────

def setup_mlflow(experiment_name: str = 'arabic-pii-ner', db_path: str = 'mlflow.db'):
    """Configure MLflow with a local SQLite backend (required by MLflow 3.x)."""
    uri = f'sqlite:///{db_path}'
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {uri}")
    print(f"View runs after training: mlflow ui --backend-store-uri {uri}")


# ── HuggingFace Trainer callback ──────────────────────────────────────────────

class MLflowEpochCallback(TrainerCallback):
    """
    Fires after each evaluation (= each epoch with eval_strategy='epoch').
    Logs metrics to both MLflow and the JSON tracker simultaneously.
    """

    def __init__(self, json_tracker: ExperimentTracker):
        self.tracker = json_tracker

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ):
        if not metrics:
            return
        epoch = int(state.epoch) if state.epoch else 0
        clean = {k.replace('eval_', ''): v for k, v in metrics.items() if isinstance(v, (int, float))}

        # JSON
        self.tracker.log_epoch(epoch, clean)

        # MLflow (safe — if no active run, skip silently)
        try:
            if mlflow.active_run():
                mlflow.log_metrics(clean, step=epoch)
        except Exception:
            pass

        # Print summary line
        f1 = clean.get('f1', 0)
        prec = clean.get('precision', 0)
        rec = clean.get('recall', 0)
        print(f"\n  [Epoch {epoch}] F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
        for ent in PII_ENTITIES:
            ent_f1 = clean.get(f'{ent}_f1', None)
            if ent_f1 is not None and clean.get(f'{ent}_support', 0) > 0:
                print(f"    {ent:20s} F1={ent_f1:.4f}  support={int(clean.get(f'{ent}_support',0))}")


# ── Seqeval evaluator ─────────────────────────────────────────────────────────

class PIIEvaluator:
    """
    Entity-level seqeval metrics for the HuggingFace Trainer's compute_metrics hook.
    Entity-level F1 (not token-level): the entire span must match to count as correct.
    """

    def __init__(self, id_to_label: Dict[int, str]):
        self.id_to_label = id_to_label

    def decode(self, predictions, labels):
        import numpy as np
        preds = np.argmax(predictions, axis=2)
        true_preds, true_labels = [], []
        for pred_seq, label_seq in zip(preds, labels):
            p_list, l_list = [], []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    p_list.append(self.id_to_label[p])
                    l_list.append(self.id_to_label[l])
            true_preds.append(p_list)
            true_labels.append(l_list)
        return true_preds, true_labels

    def compute(self, predictions, labels) -> Dict[str, float]:
        true_preds, true_labels = self.decode(predictions, labels)
        metrics = {
            'precision': precision_score(true_labels, true_preds, scheme=IOB2, zero_division=0),
            'recall':    recall_score(true_labels, true_preds, scheme=IOB2, zero_division=0),
            'f1':        f1_score(true_labels, true_preds, scheme=IOB2, zero_division=0),
        }
        report = classification_report(
            true_labels, true_preds, scheme=IOB2, output_dict=True, zero_division=0
        )
        for entity in PII_ENTITIES:
            if entity in report:
                metrics[f'{entity}_f1']     = report[entity]['f1-score']
                metrics[f'{entity}_support'] = report[entity]['support']
        return metrics

    def print_report(self, predictions, labels):
        true_preds, true_labels = self.decode(predictions, labels)
        print(classification_report(true_labels, true_preds, scheme=IOB2, zero_division=0))
