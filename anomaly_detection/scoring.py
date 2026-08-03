from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Calibration:
    mse_min: float
    mse_max: float
    cosine_min: float
    cosine_max: float
    threshold: float
    threshold_method: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Calibration":
        return cls(**values)


def cosine_distance(feature: np.ndarray, centroid: np.ndarray, epsilon: float = 1e-8) -> float:
    denominator = max(float(np.linalg.norm(feature) * np.linalg.norm(centroid)), epsilon)
    return float(1.0 - np.dot(feature, centroid) / denominator)


class AnomalyScorer:
    def __init__(self, weights: dict, calibration: Calibration, epsilon: float = 1e-8) -> None:
        self.weights = {key: float(value) for key, value in weights.items()}
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Anomaly-score weights must sum to 1.0.")
        self.calibration = calibration
        self.epsilon = epsilon

    def normalize(self, value: float, lower: float, upper: float) -> float:
        return float(np.clip((value - lower) / max(upper - lower, self.epsilon), 0.0, 1.0))

    def score(self, mse: float, cosine: float, confidence: float) -> dict[str, float]:
        mse_normalized = self.normalize(mse, self.calibration.mse_min, self.calibration.mse_max)
        cosine_normalized = self.normalize(cosine, self.calibration.cosine_min, self.calibration.cosine_max)
        confidence_normalized = float(np.clip(confidence, 0.0, 1.0))
        anomaly_score = (
            self.weights["mse"] * mse_normalized
            + self.weights["cosine"] * cosine_normalized
            + self.weights["confidence"] * (1.0 - confidence_normalized)
        )
        return {
            "mse_normalized": mse_normalized,
            "cosine_normalized": cosine_normalized,
            "confidence_normalized": confidence_normalized,
            "anomaly_score": float(anomaly_score),
        }


def fit_calibration(mse: np.ndarray, cosine: np.ndarray, confidence: np.ndarray, config: dict) -> Calibration:
    mse_min, mse_max = float(mse.min()), float(mse.max())
    cosine_min, cosine_max = float(cosine.min()), float(cosine.max())
    provisional = Calibration(mse_min, mse_max, cosine_min, cosine_max, 0.0, config["threshold"]["method"])
    scorer = AnomalyScorer(config["weights"], provisional, config["normalization"]["epsilon"])
    values = np.array([scorer.score(item_mse, item_cosine, item_confidence)["anomaly_score"] for item_mse, item_cosine, item_confidence in zip(mse, cosine, confidence)])
    threshold_config = config["threshold"]
    if threshold_config["method"] == "mean_std":
        threshold = float(values.mean() + threshold_config["std_multiplier"] * values.std())
    elif threshold_config["method"] == "percentile":
        threshold = float(np.percentile(values, threshold_config["percentile"]))
    else:
        raise ValueError("threshold.method must be 'mean_std' or 'percentile'.")
    provisional.threshold = min(1.0, threshold)
    return provisional
