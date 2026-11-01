"""Weak positive augmentation by shrinkage, linear C-SVC, and rank calibration."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _features(values):
    x = np.asarray(values, dtype=np.float64, order="F")
    if x.ndim != 2 or x.shape[1] != 54 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite (N, 54) feature matrix.")
    return x


def weak_positive_examples(cover, stego, lambdas=(0.1, 0.25, 0.5), beta=0.1):
    """Construct weighted training rows from standardized, balanced F examples."""
    cover, stego = _features(cover), _features(stego)
    if cover.shape != stego.shape:
        raise ValueError("F-cover and F-stego must have equal sample counts.")
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if (lambdas.ndim != 1 or len(lambdas) == 0 or not np.isfinite(lambdas).all()
            or np.any(lambdas <= 0) or np.any(lambdas >= 1)):
        raise ValueError("Each contraction factor must lie strictly between 0 and 1.")
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must lie strictly between 0 and 1.")
    n, k = len(cover), len(lambdas)
    center = cover.mean(axis=0)
    weak = [center + factor * (stego - center) for factor in lambdas]
    # Cover, contracted positives, and source positives form a fixed row order.
    x = np.concatenate([cover, *weak, center + (stego - center)])
    y = np.r_[np.zeros(n, dtype=np.int8), np.ones((k + 1) * n, dtype=np.int8)]
    weight = np.r_[np.ones(n), np.full(k * n, beta / k), np.full(n, 1 - beta)]
    weight *= 2 * n / weight.sum()
    return x, y, weight


class WEASDetector:
    """Fit WEA-S on real F features and calibrate using held-out C-cover features."""

    def __init__(self, lambdas=(0.1, 0.25, 0.5), beta=0.1, C=1.0, eta=0.05):
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be finite and positive.")
        if not np.isfinite(eta) or not 0 < eta < 1:
            raise ValueError("eta must lie strictly between 0 and 1.")
        self.lambdas = tuple(lambdas)
        self.beta = beta
        self.C = C
        self.eta = eta

    def fit(self, cover, stego, calibration_cover):
        """Fit scaler/SVC on F and store cover calibration scores from C."""
        cover, stego, calibration_cover = map(
            _features, (cover, stego, calibration_cover)
        )
        if cover.shape != stego.shape:
            raise ValueError("F-cover and F-stego must have equal sample counts.")
        self.scaler_ = StandardScaler().fit(np.concatenate([cover, stego]))
        x, y, weight = weak_positive_examples(
            self.scaler_.transform(cover), self.scaler_.transform(stego),
            self.lambdas, self.beta,
        )
        self.classifier_ = SVC(
            kernel="linear", C=self.C, tol=1e-3, cache_size=200,
            probability=False, max_iter=20_000_000,
        ).fit(x, y, sample_weight=weight)
        if self.classifier_.fit_status_ != 0:
            raise RuntimeError("The linear SVC did not converge.")
        self.calibration_scores_ = self.decision_function(calibration_cover)
        self._negative_calibration = np.sort(-self.calibration_scores_)
        return self

    def decision_function(self, features):
        """Return native SVC scores; larger values indicate stego."""
        if not hasattr(self, "classifier_"):
            raise RuntimeError("Fit the detector before scoring.")
        return self.classifier_.decision_function(
            self.scaler_.transform(_features(features))
        )

    def p_values(self, features):
        """Upper-tail cover ranks, including all calibration ties."""
        if not hasattr(self, "_negative_calibration"):
            raise RuntimeError("Fit and calibrate the detector before prediction.")
        scores = self.decision_function(features)
        count = np.searchsorted(self._negative_calibration, -scores, side="right")
        return (1 + count) / (len(self._negative_calibration) + 1)

    def predict(self, features):
        """Return labels: 0 for cover and 1 for stego, using p <= eta."""
        return (self.p_values(features) <= self.eta).astype(np.int8)
