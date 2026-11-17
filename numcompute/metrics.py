"""Evaluation metrics: classification, regression, ROC/AUC.

Labels may be integers, strings or booleans. Rows of a confusion matrix are
true classes, columns are predicted classes, both in sorted label order.

Streaming
---------
:class:`StreamingMetrics`, :class:`RollingMetrics` and :class:`StreamingAUC`
wrap the same formulas in an ``update`` / ``result`` / ``reset`` interface for
data that arrives in chunks. :class:`StreamingMetrics` keeps only an
accumulated confusion matrix, so its memory is O(n_classes^2) no matter how
long the stream is; the other two must retain observations and say so in
their docstrings.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import as_float_array

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "roc_curve",
    "auc",
    "roc_auc_score",
    "StreamingMetrics",
    "RollingMetrics",
    "StreamingAUC",
]

_AVERAGES = ("binary", "macro", "micro", "weighted", None)


def _as_labels(y, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    if arr.dtype.kind == "O":
        raise TypeError(f"{name} must hold numbers, strings or booleans, not objects")
    return arr


def _check_pair(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {yt.shape} and {yp.shape}")
    if yt.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return yt, yp


def _safe_div(num, den, zero_division: float) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), float(zero_division))


def accuracy(y_true, y_pred, normalize: bool = True):
    """Fraction (or count) of correct predictions.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels.
    normalize : bool, optional
        Return a fraction (default) or the number of correct predictions.

    Returns
    -------
    float or int
        In ``[0, 1]`` if ``normalize`` else in ``[0, n]``.

    Raises
    ------
    TypeError
        If labels are objects.
    ValueError
        If the inputs are not 1-D, differ in shape, are empty, or contain NaN.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    0.75
    """
    yt, yp = _check_pair(y_true, y_pred)
    correct = int(np.count_nonzero(yt == yp))
    return correct / yt.size if normalize else correct


def confusion_matrix(y_true, y_pred, labels=None, normalize: Optional[str] = None) -> np.ndarray:
    """Confusion matrix ``C[i, j]`` = samples of true class i predicted as j.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels.
    labels : array_like, optional
        Class order for rows/columns. Default: sorted union of both inputs.
        Samples whose label is not in ``labels`` are ignored.
    normalize : {None, 'true', 'pred', 'all'}, optional
        Divide by row sums, column sums, or the grand total (zero rows/columns
        stay 0).

    Returns
    -------
    numpy.ndarray, shape (n_labels, n_labels)
        Integer counts (float if normalised).

    Raises
    ------
    ValueError
        If inputs are invalid (see :func:`accuracy`), ``labels`` is not 1-D
        or contains duplicates, or ``normalize`` is invalid.

    Notes
    -----
    Built with one ``bincount`` on ``true_index * n_labels + pred_index``.
    Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1]).tolist()
    [[1, 1], [0, 2]]
    """
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError(f"normalize must be None, 'true', 'pred' or 'all', got {normalize!r}")
    yt, yp = _check_pair(y_true, y_pred)
    if labels is None:
        labs = np.unique(np.concatenate([yt, yp]))
    else:
        labs = np.asarray(labels)
        if labs.ndim != 1 or labs.size == 0:
            raise ValueError("labels must be a non-empty 1-D array")
        if np.unique(labs).size != labs.size:
            raise ValueError("labels must not contain duplicates")
    order = np.argsort(labs, kind="stable")
    sorted_labs = labs[order]

    def to_index(y):
        pos = np.minimum(np.searchsorted(sorted_labs, y), sorted_labs.size - 1)
        ok = sorted_labs[pos] == y
        return order[pos], ok

    it, ok_t = to_index(yt)
    ip, ok_p = to_index(yp)
    keep = ok_t & ok_p
    n_lab = labs.size
    cm = np.bincount(it[keep] * n_lab + ip[keep], minlength=n_lab * n_lab).reshape(n_lab, n_lab)
    if normalize is None:
        return cm
    cm = cm.astype(float)
    denom = {"true": cm.sum(axis=1, keepdims=True), "pred": cm.sum(axis=0, keepdims=True), "all": cm.sum()}[normalize]
    return np.where(denom > 0, cm / np.where(denom > 0, denom, 1.0), 0.0)


def _counts(y_true, y_pred, average, pos_label):
    """Return (labels, tp, fp, fn, support) for the chosen averaging."""
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
    yt, yp = _check_pair(y_true, y_pred)
    labels = np.unique(np.concatenate([yt, yp]))
    if average == "binary":
        if labels.dtype.kind in "US" and not isinstance(pos_label, str):
            raise ValueError(
                f"labels are strings {labels.tolist()}; pass pos_label as one of them"
            )
        labels = np.unique(np.append(labels, pos_label))
        if labels.size > 2:
            raise ValueError(
                f"average='binary' needs at most 2 classes including pos_label={pos_label!r}, "
                f"found {labels.tolist()}; set pos_label or use average='macro'/'micro'/'weighted'"
            )
    cm = confusion_matrix(yt, yp, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)
    if average == "binary":
        j = int(np.searchsorted(labels, pos_label))
        return labels, tp[j:j + 1], fp[j:j + 1], fn[j:j + 1], support[j:j + 1]
    return labels, tp, fp, fn, support


def _average(values, average, tp, fp, fn, support, kind, zero_division):
    if average == "micro":
        t, p, n = tp.sum(), fp.sum(), fn.sum()
        if kind == "precision":
            return float(_safe_div(t, t + p, zero_division))
        if kind == "recall":
            return float(_safe_div(t, t + n, zero_division))
        return float(_safe_div(2 * t, 2 * t + p + n, zero_division))
    if average == "binary":
        return float(values[0])
    if average == "macro":
        return float(values.mean())
    if average == "weighted":
        total = support.sum()
        return float((values * support).sum() / total) if total > 0 else float(zero_division)
    return values


def precision(y_true, y_pred, average: Optional[str] = "binary", pos_label=1, zero_division: float = 0.0):
    """Precision = TP / (TP + FP).

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels.
    average : {'binary', 'macro', 'micro', 'weighted', None}, optional
        ``binary``: score of ``pos_label`` only (needs at most two classes).
        ``macro``: unweighted mean over classes. ``micro``: global counts.
        ``weighted``: mean weighted by class support. ``None``: one score
        per class (sorted label order).
    pos_label : scalar, optional
        Positive class for ``average='binary'``. Default 1.
    zero_division : float, optional
        Value used when a denominator is 0. Default 0.

    Returns
    -------
    float or numpy.ndarray
        Array of shape ``(n_classes,)`` when ``average`` is None.

    Raises
    ------
    ValueError
        For invalid inputs, an unknown ``average``, or ``average='binary'``
        with more than two classes.

    Notes
    -----
    Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> precision([1, 0, 1, 1], [1, 1, 0, 1])
    0.6666666666666666
    """
    labels, tp, fp, fn, support = _counts(y_true, y_pred, average, pos_label)
    values = _safe_div(tp, tp + fp, zero_division)
    return _average(values, average, tp, fp, fn, support, "precision", zero_division)


def recall(y_true, y_pred, average: Optional[str] = "binary", pos_label=1, zero_division: float = 0.0):
    """Recall = TP / (TP + FN).

    Parameters
    ----------
    y_true, y_pred, average, pos_label, zero_division
        As in :func:`precision`.

    Returns
    -------
    float or numpy.ndarray
        As in :func:`precision`.

    Raises
    ------
    ValueError
        As in :func:`precision`.

    Notes
    -----
    Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> recall([1, 0, 1, 1], [1, 1, 0, 1])
    0.6666666666666666
    """
    labels, tp, fp, fn, support = _counts(y_true, y_pred, average, pos_label)
    values = _safe_div(tp, tp + fn, zero_division)
    return _average(values, average, tp, fp, fn, support, "recall", zero_division)


def f1(y_true, y_pred, average: Optional[str] = "binary", pos_label=1, zero_division: float = 0.0):
    """F1 score = 2 TP / (2 TP + FP + FN) (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true, y_pred, average, pos_label, zero_division
        As in :func:`precision`.

    Returns
    -------
    float or numpy.ndarray
        As in :func:`precision`.

    Raises
    ------
    ValueError
        As in :func:`precision`.

    Notes
    -----
    Written without dividing by ``precision + recall``, so it is well defined
    when both are 0. Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> f1([1, 0, 1, 1], [1, 1, 0, 1])
    0.6666666666666666
    """
    labels, tp, fp, fn, support = _counts(y_true, y_pred, average, pos_label)
    values = _safe_div(2 * tp, 2 * tp + fp + fn, zero_division)
    return _average(values, average, tp, fp, fn, support, "f1", zero_division)


def _check_regression(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    yt = as_float_array(y_true, name="y_true", allow_empty=False)
    yp = as_float_array(y_pred, name="y_pred", allow_empty=False)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {yt.shape} and {yp.shape}")
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("y_true / y_pred contain NaN")
    return yt, yp


def mse(y_true, y_pred) -> float:
    """Mean squared error.

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        Same shape (any number of dimensions), non-empty, no NaN.

    Returns
    -------
    float
        ``mean((y_true - y_pred) ** 2)`` over all elements.

    Raises
    ------
    TypeError
        If inputs are not numeric.
    ValueError
        If shapes differ, inputs are empty or contain NaN.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> mse([1, 2, 3], [1, 2, 5])
    1.3333333333333333
    """
    yt, yp = _check_regression(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (square root of :func:`mse`).

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        As in :func:`mse`.

    Returns
    -------
    float

    Raises
    ------
    TypeError, ValueError
        As in :func:`mse`.

    Notes
    -----
    Time O(n), space O(n).
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        As in :func:`mse`.

    Returns
    -------
    float
        ``mean(abs(y_true - y_pred))``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mse`.

    Notes
    -----
    Time O(n), space O(n).
    """
    yt, yp = _check_regression(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        As in :func:`mse`.

    Returns
    -------
    float
        1 is perfect; can be negative. If ``y_true`` is constant the result
        is 1.0 for a perfect fit and 0.0 otherwise.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mse`.

    Notes
    -----
    Time O(n), space O(n).
    """
    yt, yp = _check_regression(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def roc_curve(y_true, y_score, pos_label=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve for binary classification.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        Binary labels.
    y_score : array_like of float, shape (n,)
        Scores; higher means more likely positive. No NaN.
    pos_label : scalar, optional
        Label treated as positive. Default 1.

    Returns
    -------
    fpr, tpr : numpy.ndarray, shape (n_thresholds + 1,)
        False/true positive rates, starting at (0, 0) and ending at (1, 1).
    thresholds : numpy.ndarray, shape (n_thresholds + 1,)
        Decreasing; the first entry is ``inf`` (nothing predicted positive).

    Raises
    ------
    ValueError
        If shapes differ, inputs are empty/NaN, or only one class is present.

    Notes
    -----
    Tied scores share one threshold, so there is one point per distinct
    score. Time O(n log n), space O(n).

    Examples
    --------
    >>> fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr.tolist(), tpr.tolist()
    ([0.0, 0.0, 0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0, 1.0])
    """
    yt = _as_labels(y_true, "y_true")
    ys = as_float_array(y_score, name="y_score", ndim=1, allow_empty=False)
    if yt.shape != ys.shape:
        raise ValueError(f"y_true and y_score must have the same shape, got {yt.shape} and {ys.shape}")
    if np.isnan(ys).any():
        raise ValueError("y_score contains NaN")
    pos = yt == pos_label
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both classes: y_true has only one class (check pos_label)")
    order = np.argsort(-ys, kind="stable")
    s, p = ys[order], pos[order]
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    cut = np.nonzero(last_of_tie)[0]
    tps = np.cumsum(p)[cut]
    fps = (cut + 1) - tps
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    thresholds = np.r_[np.inf, s[cut]]
    return fpr, tpr, thresholds


def auc(x, y) -> float:
    """Area under a curve by the trapezoidal rule.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Curve points, ``n >= 2``. ``x`` must be monotonic (increasing or
        decreasing).

    Returns
    -------
    float
        Area (positive for increasing ``x``).

    Raises
    ------
    ValueError
        If shapes differ, fewer than 2 points, or ``x`` is not monotonic.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> auc([0, 1], [0, 1])
    0.5
    """
    xs = as_float_array(x, name="x", ndim=1)
    ys = as_float_array(y, name="y", ndim=1)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same shape, got {xs.shape} and {ys.shape}")
    if xs.size < 2:
        raise ValueError("at least 2 points are needed to compute an area")
    dx = np.diff(xs)
    if not (np.all(dx >= 0) or np.all(dx <= 0)):
        raise ValueError("x must be monotonic (increasing or decreasing)")
    direction = -1.0 if (dx < 0).any() else 1.0
    return float(direction * np.sum(dx * (ys[1:] + ys[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, pos_label=1) -> float:
    """Area under the ROC curve.

    Parameters
    ----------
    y_true, y_score, pos_label
        As in :func:`roc_curve`.

    Returns
    -------
    float
        In ``[0, 1]``; 0.5 is chance level. Ties between a positive and a
        negative score count as half.

    Raises
    ------
    ValueError
        As in :func:`roc_curve`.

    Notes
    -----
    Time O(n log n), space O(n).

    Examples
    --------
    >>> roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.75
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)
    return auc(fpr, tpr)


# --------------------------------------------------------------------------
# Streaming metrics
# --------------------------------------------------------------------------
def _scores_from_counts(tp, fp, fn, support, average, zero_division):
    """Precision, recall and F1 from per-class counts, using the batch rules."""
    if np.size(tp) == 0:  # nothing seen yet: averaging over no classes is undefined
        blank = np.zeros(0) if average is None else float(zero_division)
        return blank, blank, blank
    prec = _safe_div(tp, tp + fp, zero_division)
    rec = _safe_div(tp, tp + fn, zero_division)
    f1v = _safe_div(2 * tp, 2 * tp + fp + fn, zero_division)
    return (
        _average(prec, average, tp, fp, fn, support, "precision", zero_division),
        _average(rec, average, tp, fp, fn, support, "recall", zero_division),
        _average(f1v, average, tp, fp, fn, support, "f1", zero_division),
    )


def _check_chunk(y_true_chunk, y_pred_chunk) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a chunk pair, allowing the empty chunk that ``_check_pair`` rejects."""
    yt = _as_labels(y_true_chunk, "y_true_chunk")
    yp = _as_labels(y_pred_chunk, "y_pred_chunk")
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true_chunk and y_pred_chunk must have the same shape, got {yt.shape} and {yp.shape}"
        )
    return yt, yp


class StreamingMetrics:
    """Classification metrics accumulated over a stream of chunks.

    Only a confusion matrix is retained, so memory is O(n_classes^2) however
    many rows pass through, and every score is derived from it with the same
    formulas as the batch functions in this module. The result after feeding
    chunks therefore equals the batch result on the concatenated data.

    Parameters
    ----------
    labels : array_like, optional
        Fixed class order. Default ``None``, which discovers classes as they
        appear and grows the matrix, keeping labels sorted. With a fixed
        ``labels`` any sample whose label is not listed is ignored, exactly as
        in :func:`confusion_matrix`.
    average : {'binary', 'macro', 'micro', 'weighted', None}, optional
        Averaging for precision/recall/F1, as in :func:`precision`. Default
        ``'macro'``, which treats every class equally and is defined for any
        number of classes. ``'binary'`` requires at most two classes.
    pos_label : scalar, optional
        Positive class for ``average='binary'``. Default 1.
    zero_division : float, optional
        Value substituted when a denominator is 0, including the accuracy of
        an empty stream. Default 0.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Classes seen so far, sorted.
    n_updates_ : int
        Number of non-empty :meth:`update` calls.

    Raises
    ------
    ValueError
        If ``average`` is unknown.

    Notes
    -----
    Time O(n log L) per chunk, space O(L^2).

    Examples
    --------
    >>> m = StreamingMetrics().update([0, 1], [0, 0]).update([1, 1], [1, 1])
    >>> m.accuracy()
    0.75
    >>> m.confusion_matrix().tolist()
    [[1, 0], [1, 2]]
    """

    def __init__(
        self,
        labels=None,
        average: Optional[str] = "macro",
        pos_label=1,
        zero_division: float = 0.0,
    ) -> None:
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
        self.labels = labels
        self.average = average
        self.pos_label = pos_label
        self.zero_division = float(zero_division)
        self.reset()

    def reset(self) -> "StreamingMetrics":
        """Forget every observation seen so far.

        Returns
        -------
        StreamingMetrics
            ``self``, so calls can be chained.
        """
        if self.labels is None:
            self.labels_ = np.empty(0)
        else:
            labs = np.asarray(self.labels)
            if labs.ndim != 1 or labs.size == 0:
                raise ValueError("labels must be a non-empty 1-D array")
            if np.unique(labs).size != labs.size:
                raise ValueError("labels must not contain duplicates")
            self.labels_ = np.sort(labs)
        self._cm = np.zeros((self.labels_.size, self.labels_.size), dtype=np.int64)
        self.n_updates_ = 0
        return self

    def _grow(self, seen: np.ndarray) -> None:
        """Expand the matrix when a previously unseen class turns up."""
        merged = np.union1d(self.labels_, seen) if self.labels_.size else np.unique(seen)
        if merged.size == self.labels_.size and merged.dtype == self.labels_.dtype:
            return
        bigger = np.zeros((merged.size, merged.size), dtype=np.int64)
        if self.labels_.size:
            pos = np.searchsorted(merged, self.labels_)
            bigger[np.ix_(pos, pos)] = self._cm
        self.labels_, self._cm = merged, bigger

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingMetrics":
        """Fold one chunk of predictions into the confusion matrix.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array_like, shape (n,)
            True and predicted labels for this chunk. An empty chunk is
            accepted and changes nothing, because a stream may deliver one.

        Returns
        -------
        StreamingMetrics
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If the labels have object dtype.
        ValueError
            If the inputs are not 1-D, differ in shape, or contain NaN.

        Notes
        -----
        Classes missing from a chunk need no special handling: the matrix
        simply gains zeros in their row and column. Time O(n log L), space
        O(n + L^2).
        """
        yt, yp = _check_chunk(y_true_chunk, y_pred_chunk)
        if yt.size == 0:
            return self
        if self.labels is None:
            self._grow(np.unique(np.concatenate([yt, yp])))
        self._cm += confusion_matrix(yt, yp, labels=self.labels_)
        self.n_updates_ += 1
        return self

    @property
    def n_samples_(self) -> int:
        """Samples counted in the matrix (excludes labels outside ``labels``)."""
        return int(self._cm.sum())

    def confusion_matrix(self) -> np.ndarray:
        """Accumulated confusion matrix, shape ``(n_classes, n_classes)``."""
        return self._cm.copy()

    def accuracy(self) -> float:
        """Fraction of correct predictions over everything accumulated."""
        total = self._cm.sum()
        if total == 0:
            return self.zero_division
        return float(np.trace(self._cm) / total)

    def _counts(self):
        cm = self._cm.astype(float)
        tp = np.diag(cm).copy()
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        support = cm.sum(axis=1)
        if self.average == "binary":
            if self.labels_.size > 2:
                raise ValueError(
                    f"average='binary' needs at most 2 classes, found {self.labels_.tolist()}; "
                    f"use average='macro', 'micro' or 'weighted'"
                )
            j = int(np.searchsorted(self.labels_, self.pos_label))
            if j >= self.labels_.size or self.labels_[j] != self.pos_label:
                return (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))
            return tp[j:j + 1], fp[j:j + 1], fn[j:j + 1], support[j:j + 1]
        return tp, fp, fn, support

    def precision(self):
        """Precision under the configured ``average``."""
        return _scores_from_counts(*self._counts(), self.average, self.zero_division)[0]

    def recall(self):
        """Recall under the configured ``average``."""
        return _scores_from_counts(*self._counts(), self.average, self.zero_division)[1]

    def f1(self):
        """F1 score under the configured ``average``."""
        return _scores_from_counts(*self._counts(), self.average, self.zero_division)[2]

    def result(self) -> Dict[str, Any]:
        """Every accumulated metric in one dictionary.

        Returns
        -------
        dict
            Keys ``n_samples``, ``n_updates``, ``labels``, ``accuracy``,
            ``error_rate``, ``precision``, ``recall``, ``f1`` and
            ``confusion_matrix``.

        Notes
        -----
        Time O(L^2), space O(L^2).
        """
        prec, rec, f1v = _scores_from_counts(*self._counts(), self.average, self.zero_division)
        acc = self.accuracy()
        return {
            "n_samples": self.n_samples_,
            "n_updates": self.n_updates_,
            "labels": self.labels_.copy(),
            "accuracy": acc,
            "error_rate": 1.0 - acc if self.n_samples_ else self.zero_division,
            "precision": prec,
            "recall": rec,
            "f1": f1v,
            "confusion_matrix": self.confusion_matrix(),
        }

    def __repr__(self) -> str:
        return (
            f"StreamingMetrics(average={self.average!r}, n_samples_={self.n_samples_}, "
            f"n_classes={self.labels_.size})"
        )


class RollingMetrics:
    """Classification metrics over the most recent ``window`` observations.

    Unlike :class:`StreamingMetrics`, which summarises the whole stream, this
    answers "how is the model doing *now*", which is what reveals that a tree
    has gone stale between retraining events. It has to retain the window, so
    memory is O(window); that is inherent to a rolling statistic rather than
    an implementation shortcut.

    Parameters
    ----------
    window : int
        Number of most recent samples to keep, ``>= 1``.
    labels, average, pos_label, zero_division
        As in :class:`StreamingMetrics`.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``average`` is unknown.

    Notes
    -----
    Time O(window log L) per :meth:`result`, space O(window).

    Examples
    --------
    >>> r = RollingMetrics(window=2).update([1, 1, 0], [0, 0, 0])
    >>> r.result()["n_samples"], r.accuracy()      # only the last two count
    (2, 0.5)
    """

    def __init__(
        self,
        window: int,
        labels=None,
        average: Optional[str] = "macro",
        pos_label=1,
        zero_division: float = 0.0,
    ) -> None:
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
            raise ValueError(f"window must be an int >= 1, got {window!r}")
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
        self.window = int(window)
        self.labels = labels
        self.average = average
        self.pos_label = pos_label
        self.zero_division = float(zero_division)
        self.reset()

    def reset(self) -> "RollingMetrics":
        """Empty the window.

        Returns
        -------
        RollingMetrics
            ``self``, so calls can be chained.
        """
        self._true: deque = deque(maxlen=self.window)
        self._pred: deque = deque(maxlen=self.window)
        return self

    def update(self, y_true_chunk, y_pred_chunk) -> "RollingMetrics":
        """Append a chunk, dropping whatever falls out of the window.

        Parameters
        ----------
        y_true_chunk, y_pred_chunk : array_like, shape (n,)
            True and predicted labels. An empty chunk changes nothing.

        Returns
        -------
        RollingMetrics
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError, ValueError
            As in :meth:`StreamingMetrics.update`.

        Notes
        -----
        A chunk longer than the window keeps only its last ``window`` rows.
        Time O(n), space O(window).
        """
        yt, yp = _check_chunk(y_true_chunk, y_pred_chunk)
        if yt.size == 0:
            return self
        self._true.extend(yt.tolist())
        self._pred.extend(yp.tolist())
        return self

    def _snapshot(self) -> StreamingMetrics:
        snap = StreamingMetrics(
            labels=self.labels,
            average=self.average,
            pos_label=self.pos_label,
            zero_division=self.zero_division,
        )
        return snap.update(np.asarray(self._true), np.asarray(self._pred))

    @property
    def n_samples_(self) -> int:
        """Observations currently inside the window."""
        return len(self._true)

    def accuracy(self) -> float:
        """Accuracy over the window."""
        return self._snapshot().accuracy()

    def confusion_matrix(self) -> np.ndarray:
        """Confusion matrix over the window."""
        return self._snapshot().confusion_matrix()

    def result(self) -> Dict[str, Any]:
        """Same keys as :meth:`StreamingMetrics.result`, restricted to the window."""
        out = self._snapshot().result()
        out["window"] = self.window
        return out

    def __repr__(self) -> str:
        return f"RollingMetrics(window={self.window}, n_samples_={self.n_samples_})"


class StreamingAUC:
    """Area under the ROC curve accumulated over a stream.

    **All scores are retained.** AUC compares every positive against every
    negative, so unlike accuracy or F1 it cannot be recovered from a fixed
    number of counters; the exact value needs the scores themselves. This
    class therefore stores them and computes the exact AUC with
    :func:`roc_auc_score`, at a memory cost of O(n_seen). Call :meth:`reset`
    periodically, or use ``max_scores`` to keep only the most recent scores,
    when that cost matters.

    Parameters
    ----------
    pos_label : scalar, optional
        Label treated as positive. Default 1.
    max_scores : int, optional
        Keep only the most recent ``max_scores`` observations. Default
        ``None`` (keep everything, exact over the whole stream). When set,
        the result is the exact AUC of the retained suffix, not of the whole
        stream, and :attr:`n_dropped_` records how many were discarded.

    Attributes
    ----------
    n_seen_ : int
        Observations passed to :meth:`update`.
    n_dropped_ : int
        Observations discarded because of ``max_scores``.

    Raises
    ------
    ValueError
        If ``max_scores`` is not a positive integer.

    Notes
    -----
    Time O(n log n) per :meth:`result`, space O(min(n_seen, max_scores)).

    Examples
    --------
    >>> a = StreamingAUC().update([0, 0], [0.1, 0.4]).update([1, 1], [0.35, 0.8])
    >>> a.result()
    0.75
    """

    def __init__(self, pos_label=1, max_scores: Optional[int] = None) -> None:
        if max_scores is not None:
            if isinstance(max_scores, bool) or not isinstance(max_scores, (int, np.integer)) or max_scores < 1:
                raise ValueError(f"max_scores must be None or an int >= 1, got {max_scores!r}")
            max_scores = int(max_scores)
        self.pos_label = pos_label
        self.max_scores = max_scores
        self.reset()

    def reset(self) -> "StreamingAUC":
        """Discard every retained score.

        Returns
        -------
        StreamingAUC
            ``self``, so calls can be chained.
        """
        self._true: deque = deque(maxlen=self.max_scores)
        self._score: deque = deque(maxlen=self.max_scores)
        self.n_seen_ = 0
        self.n_dropped_ = 0
        return self

    def update(self, y_true_chunk, y_score_chunk) -> "StreamingAUC":
        """Retain one chunk of labels and scores.

        Parameters
        ----------
        y_true_chunk : array_like, shape (n,)
            True labels.
        y_score_chunk : array_like of float, shape (n,)
            Scores where higher means more likely positive, e.g. a column of
            ``predict_proba``. An empty chunk changes nothing.

        Returns
        -------
        StreamingAUC
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If the shapes differ or a score is NaN.

        Notes
        -----
        Time O(n), space O(n).
        """
        yt = _as_labels(y_true_chunk, "y_true_chunk")
        ys = as_float_array(y_score_chunk, name="y_score_chunk", ndim=1)
        if yt.shape != ys.shape:
            raise ValueError(
                f"y_true_chunk and y_score_chunk must have the same shape, got {yt.shape} and {ys.shape}"
            )
        if ys.size == 0:
            return self
        if np.isnan(ys).any():
            raise ValueError("y_score_chunk contains NaN")
        before = len(self._true)
        self._true.extend(yt.tolist())
        self._score.extend(ys.tolist())
        self.n_seen_ += int(ys.size)
        self.n_dropped_ += max(0, before + int(ys.size) - len(self._true))
        return self

    def result(self) -> float:
        """Exact AUC over the retained observations.

        Returns
        -------
        float
            ``nan`` while only one class has been seen, because a ROC curve
            is undefined then. This is deliberate: early chunks of a stream
            are often single-class, and returning ``nan`` lets a caller plot
            the metric without special-casing the warm-up.

        Notes
        -----
        Time O(n log n), space O(n).
        """
        if len(self._true) == 0:
            return float("nan")
        yt = np.asarray(self._true)
        pos = int(np.count_nonzero(yt == self.pos_label))
        if pos == 0 or pos == yt.size:
            return float("nan")
        return roc_auc_score(yt, np.asarray(self._score), pos_label=self.pos_label)

    def __repr__(self) -> str:
        return f"StreamingAUC(n_seen_={self.n_seen_}, retained={len(self._true)})"
