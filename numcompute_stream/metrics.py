"""Streaming classification metrics.

Re-exports the metrics layer of :mod:`numcompute.metrics`:
:class:`StreamingMetrics` (confusion matrix, accuracy, precision, recall and
F1 accumulated from counts alone), :class:`RollingMetrics` (the same over a
fixed window of recent samples), :class:`StreamingAUC`, and the batch
functions they are checked against.
"""
from __future__ import annotations

from numcompute.metrics import (
    RollingMetrics,
    StreamingAUC,
    StreamingMetrics,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "StreamingMetrics",
    "RollingMetrics",
    "StreamingAUC",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "roc_curve",
    "auc",
    "roc_auc_score",
]
