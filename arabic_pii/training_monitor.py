"""
Training monitoring and experiment tracking.
Follows akschneider1/arabic-pii-ner-model training_monitor.py methodology.
Replaces wandb with local JSON logging (CPU-friendly, no API key needed).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


PII_ENTITIES = ['PERSON', 'LOCATION', 'ORGANIZATION', 'PHONE', 'EMAIL', 'ID_NUMBER', 'ADDRESS']


class ExperimentTracker:
    """
    Lightweight JSON-based experiment tracker.
    Records hyperparameters, per-step loss, and per-epoch eval metrics
    to a single JSON file — one entry per training run.
    """

    def __init__(self, log_dir: str = 'experiment_logs', run_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.run_name = run_name or f'run_{int(time.time())}'
        self.log_path = self.log_dir / f'{self.run_name}.json'

        self._record: Dict = {
            'run_name': self.run_name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': {},
            'steps': [],
            'epochs': [],
        }
        print(f"Experiment tracker: {self.log_path}")

    def log_hyperparameters(self, params: Dict):
        self._record['hyperparameters'] = params
        self._flush()

    def log_step(self, step: int, loss: float):
        self._record['steps'].append({'step': step, 'loss': round(loss, 6)})
        self._flush()

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        entry = {'epoch': epoch, **{k: round(v, 4) for k, v in metrics.items()}}
        self._record['epochs'].append(entry)
        self._flush()
        print(f"  Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def finish(self, final_metrics: Optional[Dict] = None):
        self._record['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if final_metrics:
            self._record['final_metrics'] = {k: round(v, 4) for k, v in final_metrics.items()}
        self._flush()
        print(f"Experiment saved → {self.log_path}")

    def _flush(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self._record, f, ensure_ascii=False, indent=2)


class PIIEvaluator:
    """
    Computes seqeval metrics for PII NER evaluation.
    Returns overall F1 + per-entity breakdown.
    """

    def __init__(self, id_to_label: Dict[int, str]):
        self.id_to_label = id_to_label

    def decode(self, predictions, labels) -> tuple:
        """Convert raw model output to list-of-tag-lists."""
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
            'recall': recall_score(true_labels, true_preds, scheme=IOB2, zero_division=0),
            'f1': f1_score(true_labels, true_preds, scheme=IOB2, zero_division=0),
        }

        report = classification_report(
            true_labels, true_preds, scheme=IOB2, output_dict=True, zero_division=0
        )
        for entity in PII_ENTITIES:
            if entity in report:
                metrics[f'{entity}_f1'] = report[entity]['f1-score']
                metrics[f'{entity}_support'] = report[entity]['support']

        return metrics

    def print_report(self, predictions, labels):
        true_preds, true_labels = self.decode(predictions, labels)
        print(classification_report(true_labels, true_preds, scheme=IOB2, zero_division=0))
