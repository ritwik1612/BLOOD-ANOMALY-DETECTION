from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.metrics import evaluate_anomaly_predictions, evaluate_detector
from utils.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector and, when labeled data is supplied, open-set anomaly scores.")
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--anomaly-csv", type=Path, help="CSV with anomaly_score, threshold, and ground_truth_abnormal columns.")
    args = parser.parse_args()
    results: dict[str, float] = {}
    weights = PROJECT_ROOT / "outputs" / "weights" / "yolov8n_txl_pbc_best.pt"
    data_yaml = PROJECT_ROOT / "dataset" / "processed" / "data.yaml"
    if not args.skip_detector:
        results.update(evaluate_detector(weights, data_yaml))
    if args.anomaly_csv:
        results.update(evaluate_anomaly_predictions(pd.read_csv(args.anomaly_csv), PROJECT_ROOT / "outputs" / "metrics"))
    output = PROJECT_ROOT / "outputs" / "metrics" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
