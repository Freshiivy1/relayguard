"""Platt calibration for detector scores (C5).

Each detector emits an uncalibrated P(relay) in [0,1]. PlattCalibrator fits a
1-D logistic regression (Platt scaling) mapping raw score -> calibrated
probability. CalibratorBank keeps one calibrator per detector name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """1-D Platt scaler: calibrated_p = sigmoid(a * score + b).

    Uses sklearn LogisticRegression on the single feature ``score``. If the
    training labels contain a single class, the calibrator degenerates to a
    constant predictor (mean label).
    """

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None
        self._constant: float | None = None
        self.n_fit: int = 0

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "PlattCalibrator":
        scores = np.asarray(list(scores), dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(list(labels), dtype=np.int64)
        if scores.shape[0] == 0:
            raise ValueError("PlattCalibrator.fit: empty training data")
        self.n_fit = int(scores.shape[0])
        uniq = np.unique(labels)
        if uniq.size < 2:
            # Degenerate: single-class training data -> constant predictor.
            self._lr = None
            self._constant = float(np.clip(labels.mean(), 1e-4, 1.0 - 1e-4))
            return self
        lr = LogisticRegression(max_iter=1000)
        lr.fit(scores, labels)
        self._lr = lr
        self._constant = None
        return self

    def predict(self, score: float) -> float:
        """Map a raw detector score to a calibrated probability in [0,1]."""
        if self._constant is not None:
            return self._constant
        if self._lr is None:
            return float(np.clip(score, 0.0, 1.0))  # unfitted -> identity
        proba = self._lr.predict_proba([[float(score)]])[0]
        classes = list(self._lr.classes_)
        p = proba[classes.index(1)] if 1 in classes else float(proba[-1])
        return float(np.clip(p, 0.0, 1.0))

    def predict_many(self, scores: Iterable[float]) -> np.ndarray:
        return np.array([self.predict(s) for s in scores], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a PlattCalibrator")
        return obj


class CalibratorBank:
    """Per-detector-name collection of PlattCalibrators."""

    def __init__(self, calibrators: dict[str, PlattCalibrator] | None = None) -> None:
        self.calibrators: dict[str, PlattCalibrator] = dict(calibrators or {})

    def fit_bank(self, score_rows: list[dict], labels: Iterable[int]) -> "CalibratorBank":
        """Fit one calibrator per detector name.

        score_rows: list of dicts mapping detector name -> raw score. Rows may
        be missing detectors; each calibrator is fit on the rows where that
        detector is present.
        """
        labels = list(labels)
        if len(score_rows) != len(labels):
            raise ValueError("score_rows and labels length mismatch")
        names: set[str] = set()
        for row in score_rows:
            names.update(row.keys())
        for name in sorted(names):
            xs = [row[name] for row in score_rows if name in row]
            ys = [lab for row, lab in zip(score_rows, labels) if name in row]
            self.calibrators[name] = PlattCalibrator().fit(xs, ys)
        return self

    def calibrate(self, name: str, score: float) -> float:
        """Calibrate a score; identity mapping if no calibrator for ``name``."""
        cal = self.calibrators.get(name)
        if cal is None:
            return float(np.clip(score, 0.0, 1.0))
        return cal.predict(score)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "CalibratorBank":
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a CalibratorBank")
        return obj
