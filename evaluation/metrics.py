from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from ultralytics import YOLO


def evaluate_detector(weights: Path, data_yaml: Path) -> dict[str, float]:
    metrics = YOLO(str(weights)).val(data=str(data_yaml), split="test", plots=True, verbose=False)
    precision, recall = float(metrics.box.mp), float(metrics.box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def evaluate_anomaly_predictions(predictions: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    required = {"anomaly_score", "threshold", "ground_truth_abnormal"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing columns: {sorted(missing)}")
    truth = predictions["ground_truth_abnormal"].astype(int)
    if truth.nunique() < 2:
        raise ValueError("ROC-AUC needs both normal (0) and abnormal (1) ground-truth examples.")
    scores = predictions["anomaly_score"].astype(float)
    predicted = (scores > predictions["threshold"].astype(float)).astype(int)
    fpr, tpr, _ = roc_curve(truth, scores)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(truth, scores):.3f}")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Open-set anomaly ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "anomaly_roc_curve.png", dpi=160)
    plt.close(figure)
    return {
        "anomaly_precision": float(precision_score(truth, predicted, zero_division=0)),
        "anomaly_recall": float(recall_score(truth, predicted, zero_division=0)),
        "anomaly_f1": float(f1_score(truth, predicted, zero_division=0)),
        "anomaly_roc_auc": float(roc_auc_score(truth, scores)),
    }
