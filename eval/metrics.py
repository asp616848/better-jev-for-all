"""
Calibration + accuracy metrics shared by every eval path (encoder, decoder-LoRA,
and later the cascade). Implements exactly what PRD.md Section 5.3/8.1 commits
to reporting: accuracy, multi-class Brier score, and ECE (equal-mass binning,
the standard that avoids the equal-width-binning pitfall on skewed confidence
distributions most models produce).
"""

import numpy as np


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=-1)
    return float((preds == labels).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Multi-class Brier score: mean squared error between predicted probs and one-hot labels."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE using the *confidence of the predicted class* against equal-mass bins
    (sorted-then-split), which is more robust than equal-width bins when a
    model's confidence distribution is heavily skewed toward high confidence
    (the common case post-fine-tuning) — equal-width bins would leave most
    bins near-empty and the ECE estimate noisy.
    """
    confidences = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(np.float64)

    order = np.argsort(confidences)
    confidences, correct = confidences[order], correct[order]

    n = len(confidences)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        bin_conf = confidences[lo:hi].mean()
        bin_acc = correct[lo:hi].mean()
        weight = (hi - lo) / n
        ece += weight * abs(bin_conf - bin_acc)
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200, lr: float = 0.01) -> float:
    """
    Post-hoc temperature scaling (Guo et al. 2017): find scalar T minimizing
    NLL of softmax(logits / T) on a held-out calibration split. Single
    parameter, convex in log(T), optimized here with plain gradient descent
    to avoid pulling in torch/scipy-specific optimizers for a 1D problem.
    """
    log_t = 0.0  # T starts at 1.0
    labels = labels.astype(np.int64)
    n = len(labels)
    idx = np.arange(n)

    def grad_at(log_t_val):
        t = np.exp(log_t_val)
        scaled = logits / t
        scaled -= scaled.max(axis=-1, keepdims=True)
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)

        # d(NLL)/d(1/T) = mean(E_p[z] - z_true); d(1/T)/d(log T) = -1/T.
        true_logit = logits[idx, labels]
        expected_logit = (probs * logits).sum(axis=-1)
        d_nll_d_invT = (expected_logit - true_logit).mean()
        return d_nll_d_invT * (-1.0 / t)

    for _ in range(max_iter):
        log_t -= lr * grad_at(log_t)

    return float(np.exp(log_t))


def full_report(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    return {
        "n": int(len(labels)),
        "accuracy": accuracy(probs, labels),
        "brier": brier_score(probs, labels, n_classes),
        "ece": expected_calibration_error(probs, labels),
    }
