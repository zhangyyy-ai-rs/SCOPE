"""Prototype geometry diagnostics computed from support data only."""
from __future__ import annotations

import numpy as np

from .metrics import normalize


def prototype_stats(
    train_cls: np.ndarray,
    train_y: np.ndarray,
    probe_cls: np.ndarray | None = None,
    probe_y: np.ndarray | None = None,
) -> dict:
    """Compute prototype diagnostics without reading task result files.

    `probe_cls/probe_y` may be a held-out subset carved from the support split.
    When omitted, the support split itself is used. In either case, the value is
    a support diagnostic and not a validation/test accuracy used for model
    selection.
    """
    x = normalize(np.asarray(train_cls, dtype=np.float64))
    y = np.asarray(train_y)
    xp = normalize(np.asarray(probe_cls if probe_cls is not None else train_cls, dtype=np.float64))
    yp = np.asarray(probe_y if probe_y is not None else train_y)
    classes = sorted(np.unique(y).tolist())
    protos, counts, within = [], [], []
    for c in classes:
        m = x[y == c]
        if m.size == 0:
            continue
        p = normalize(m.mean(axis=0, keepdims=True))[0]
        protos.append(p)
        counts.append(m.shape[0])
        within.append(float(np.mean(1.0 - m @ p)))
    proto = np.stack(protos, axis=0)
    sim = proto @ proto.T
    off = sim[~np.eye(len(protos), dtype=bool)] if len(protos) > 1 else np.array([0.0])
    logits = xp @ proto.T
    pred = np.asarray(classes)[logits.argmax(axis=1)]
    if logits.shape[1] > 1:
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = float(np.mean(top2[:, 1] - top2[:, 0]))
    else:
        margin = float('nan')
    within_cos_dist = float(np.mean(within))
    max_inter = float(off.max())
    return {
        'num_classes_seen': int(len(classes)),
        'prototype_inter_cos_mean': float(off.mean()),
        'prototype_inter_cos_max': max_inter,
        'within_cos_dist': within_cos_dist,
        'prototype_risk_index': within_cos_dist + max_inter,
        'nearest_centroid_support_acc': float(np.mean(pred == yp) * 100.0),
        'nearest_centroid_margin': margin,
        'class_imbalance_cv': float(np.std(counts) / (np.mean(counts) + 1e-8)),
    }
