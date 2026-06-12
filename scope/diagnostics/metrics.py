"""Numerical support-set diagnostic metrics."""
from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def effective_rank(features: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] <= 1:
        return 1.0, 1.0, 1.0
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov).clip(min=1e-12)
    prob = evals / evals.sum()
    erank = float(np.exp(-(prob * np.log(prob)).sum()))
    return erank, float(evals[-1] / evals.sum()), float(evals[-min(10, evals.size):].sum() / evals.sum())


def image_high_frequency(images_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(images_nchw, dtype=np.float32)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    grad = 0.5 * (np.mean(dx * dx, axis=(1, 2, 3)) + np.mean(dy * dy, axis=(1, 2, 3)))
    gray = x.mean(axis=1)
    fft = np.abs(np.fft.rfft2(gray, norm='ortho')) ** 2
    h, w = gray.shape[-2:]
    fy = np.abs(np.fft.fftfreq(h)).reshape(h, 1)
    fx = np.abs(np.fft.rfftfreq(w)).reshape(1, w // 2 + 1)
    radius = np.sqrt(fx * fx + fy * fy)
    high = fft[:, radius > 0.25].sum(axis=1)
    total = fft.reshape(fft.shape[0], -1).sum(axis=1).clip(min=1e-8)
    return grad, high / total
