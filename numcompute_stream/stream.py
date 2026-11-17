"""Driving a model over a stream of chunks, and logging what happens.

:class:`StreamTrainer` owns the loop that a streaming experiment repeats:
take the next chunk, evaluate the current model on it, train on it, write
down what happened. Keeping that loop in one place is what makes the demo
notebook and the buffer-size benchmark the same five lines with different
arguments.

Evaluation order
----------------
A chunk is scored *before* it is trained on. This is prequential
(test-then-train) evaluation, the standard protocol for streams: every chunk
is unseen test data at the moment it is measured, so the reported accuracy
is an honest estimate of how the model generalises rather than a reflection
of data it has already absorbed. :meth:`StreamTrainer.process_chunk` enforces
that order; calling :meth:`score_chunk` and :meth:`fit_chunk` by hand leaves
it to the caller.
"""
from __future__ import annotations

import csv
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from numcompute.metrics import RollingMetrics, StreamingAUC, StreamingMetrics
from numcompute.pipeline import Pipeline
from numcompute.utils import NotFittedError

__all__ = ["StreamTrainer"]

#: Fields written for every chunk, in log order.
_FIELDS = (
    "chunk",
    "n_samples",
    "n_samples_seen",
    "predicted",
    "accuracy",
    "error_rate",
    "cumulative_accuracy",
    "rolling_accuracy",
    "auc",
    "predict_seconds",
    "train_seconds",
    "retrain_seconds",
    "retrained",
    "n_retrains",
    "buffered",
    "memory_bytes",
)


def _final_estimator(model):
    """The object that actually learns: a Pipeline's last step, or the model."""
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def _memory_bytes(model) -> Optional[int]:
    """Bytes owned by the model, summed over pipeline steps that report it."""
    parts = [s for _, s in model.steps] if isinstance(model, Pipeline) else [model]
    sizes = [int(p.memory_bytes()) for p in parts if hasattr(p, "memory_bytes")]
    return sum(sizes) if sizes else None


class StreamTrainer:
    """Train and evaluate a model chunk by chunk, recording every step.

    Parameters
    ----------
    model : object
        A :class:`numcompute.pipeline.Pipeline` or a bare estimator. It must
        provide ``partial_fit(X, y)`` and ``predict(X)``; ``predict_proba``
        is used as well when ``track_auc`` is on.
    window : int, optional
        Length of the rolling-accuracy window, in samples. Default ``None``,
        which skips the rolling metric. A window shows how the model is doing
        *now*, which is what exposes a tree going stale between rebuilds,
        whereas cumulative accuracy is dominated by the past.
    track_auc : bool, optional
        Also accumulate a streaming AUC from ``predict_proba``. Default
        False. Only meaningful for binary targets, and it retains scores —
        see :class:`numcompute.metrics.StreamingAUC`.
    pos_label : scalar, optional
        Positive class for the AUC. Default 1.
    average : {'binary', 'macro', 'micro', 'weighted', None}, optional
        Averaging for the accumulated precision/recall/F1. Default
        ``'macro'``, which is defined for any number of classes.

    Attributes
    ----------
    history : list of dict
        One record per completed chunk; see :attr:`FIELDS` for the keys.
    metrics : numcompute.metrics.StreamingMetrics
        Cumulative metrics over every chunk that has been scored.
    rolling : numcompute.metrics.RollingMetrics or None
        Rolling metrics, when ``window`` was given.
    auc : numcompute.metrics.StreamingAUC or None
        Streaming AUC, when ``track_auc`` was set.
    n_chunks_ : int
        Chunks completed.
    n_samples_seen_ : int
        Rows passed to :meth:`fit_chunk`.

    Raises
    ------
    TypeError
        If ``model`` cannot be trained incrementally or cannot predict.

    Notes
    -----
    The three accuracies in the log are different quantities and are kept
    deliberately distinct:

    ``accuracy``
        Correct fraction on this chunk alone, before training on it.
    ``cumulative_accuracy``
        Correct fraction over every chunk scored so far. Computed from the
        accumulated confusion matrix — total correct divided by total seen —
        not by averaging the per-chunk accuracies, which would weight a short
        final chunk the same as a full one.
    ``rolling_accuracy``
        Correct fraction over the last ``window`` samples.

    Examples
    --------
    >>> import numpy as np
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X, y = rng.normal(size=(300, 3)), rng.integers(0, 2, 300)
    >>> trainer = StreamTrainer(DecisionTreeClassifier(max_depth=3, buffer_size=100))
    >>> for i in range(0, 300, 50):
    ...     _ = trainer.process_chunk(X[i:i + 50], y[i:i + 50])
    >>> len(trainer.history), trainer.n_samples_seen_
    (6, 300)
    """

    #: Keys present in every history record.
    FIELDS = _FIELDS

    def __init__(
        self,
        model,
        window: Optional[int] = None,
        track_auc: bool = False,
        pos_label=1,
        average: Optional[str] = "macro",
    ) -> None:
        if not hasattr(model, "partial_fit"):
            raise TypeError(
                f"{type(model).__name__} has no partial_fit; StreamTrainer trains "
                f"incrementally, so the model must support it"
            )
        if not hasattr(model, "predict"):
            raise TypeError(f"{type(model).__name__} has no predict")
        if track_auc and not hasattr(model, "predict_proba"):
            raise TypeError(
                f"track_auc needs predict_proba, which {type(model).__name__} does not have"
            )
        self.model = model
        self.window = window
        self.track_auc = bool(track_auc)
        self.pos_label = pos_label
        self.average = average
        self.metrics = StreamingMetrics(average=average, pos_label=pos_label)
        self.rolling = RollingMetrics(window, average=average, pos_label=pos_label) if window else None
        self.auc = StreamingAUC(pos_label=pos_label) if track_auc else None
        self.history: List[Dict[str, Any]] = []
        self.n_chunks_ = 0
        self.n_samples_seen_ = 0
        self._current: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------- bookkeeping
    def _record(self) -> Dict[str, Any]:
        """The record being filled in for the chunk in progress."""
        if self._current is None:
            self._current = {k: None for k in _FIELDS}
            self._current["chunk"] = self.n_chunks_ + 1
        return self._current

    def _retrain_count(self) -> int:
        return int(getattr(_final_estimator(self.model), "n_retrains_", 0))

    # ------------------------------------------------------------- the loop
    def score_chunk(self, X, y) -> Dict[str, Any]:
        """Evaluate the current model on one chunk, without training on it.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk features.
        y : array_like, shape (n_samples,)
            True labels for the chunk.

        Returns
        -------
        dict
            A copy of the record for the chunk in progress, with the
            evaluation fields filled in.

        Notes
        -----
        **Before the first successful training** the model has nothing to
        predict with. That is not an error in a stream, it is the warm-up, so
        the record gets ``predicted=False`` and ``accuracy=None``, and the
        cumulative metrics are left alone rather than being polluted with
        guesses. Time O(cost of predict), space O(n_samples).
        """
        rec = self._record()
        rec["n_samples"] = int(np.asarray(y).shape[0])
        started = time.perf_counter()
        try:
            y_pred = self.model.predict(X)
        except NotFittedError:
            rec["predicted"] = False
            rec["predict_seconds"] = time.perf_counter() - started
            rec["cumulative_accuracy"] = self.metrics.accuracy() if self.metrics.n_samples_ else None
            return dict(rec)
        rec["predict_seconds"] = time.perf_counter() - started
        rec["predicted"] = True

        self.metrics.update(y, y_pred)
        chunk = StreamingMetrics(average=self.average, pos_label=self.pos_label).update(y, y_pred)
        rec["accuracy"] = chunk.accuracy()
        rec["error_rate"] = 1.0 - chunk.accuracy()
        rec["cumulative_accuracy"] = self.metrics.accuracy()
        if self.rolling is not None:
            self.rolling.update(y, y_pred)
            rec["rolling_accuracy"] = self.rolling.accuracy()
        if self.auc is not None:
            proba = self.model.predict_proba(X)
            classes = np.asarray(_final_estimator(self.model).classes_)
            hit = np.nonzero(classes == self.pos_label)[0]
            if hit.size:
                self.auc.update(y, proba[:, int(hit[0])])
            rec["auc"] = self.auc.result()
        return dict(rec)

    def fit_chunk(self, X, y) -> Dict[str, Any]:
        """Train on one chunk and close its record.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk features.
        y : array_like, shape (n_samples,)
            Labels for the chunk.

        Returns
        -------
        dict
            The completed record, which is also appended to :attr:`history`.

        Raises
        ------
        Exception
            Whatever the model raises. The record is not appended and the
            chunk counter does not advance, so a failed chunk cannot leave a
            half-written entry in the log.

        Notes
        -----
        The call goes to ``model.partial_fit``, so with a buffered tree most
        chunks only append to the buffer; ``retrained`` says whether this
        chunk actually triggered a rebuild, and ``retrain_seconds`` is the
        rebuild's own time as measured by the tree, which is a subset of
        ``train_seconds``. Time O(cost of partial_fit).
        """
        rec = self._record()
        if rec["n_samples"] is None:
            rec["n_samples"] = int(np.asarray(y).shape[0])
        before = self._retrain_count()
        started = time.perf_counter()
        self.model.partial_fit(X, y)
        rec["train_seconds"] = time.perf_counter() - started

        model = _final_estimator(self.model)
        after = self._retrain_count()
        rec["retrained"] = after > before
        rec["retrain_seconds"] = float(getattr(model, "last_fit_seconds_", 0.0)) if after > before else 0.0
        rec["n_retrains"] = after
        rec["buffered"] = int(getattr(model, "n_buffered_", 0))
        rec["memory_bytes"] = _memory_bytes(self.model)
        self.n_samples_seen_ += int(rec["n_samples"])
        rec["n_samples_seen"] = self.n_samples_seen_
        self.history.append(rec)
        self.n_chunks_ += 1
        self._current = None
        return dict(rec)

    def process_chunk(self, X, y) -> Dict[str, Any]:
        """Score then train on one chunk: the prequential order.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk features.
        y : array_like, shape (n_samples,)
            Labels for the chunk.

        Returns
        -------
        dict
            The completed record for this chunk.

        Notes
        -----
        Equivalent to :meth:`score_chunk` followed by :meth:`fit_chunk`, in
        that order, which is what keeps the evaluation leakage-free.
        """
        self.score_chunk(X, y)
        return self.fit_chunk(X, y)

    def run(self, chunks: Iterable) -> List[Dict[str, Any]]:
        """Run :meth:`process_chunk` over an iterable of ``(X, y)`` pairs.

        Parameters
        ----------
        chunks : iterable of (array_like, array_like)
            For example the output of
            :func:`numcompute_stream.io.iter_chunks` with a target.

        Returns
        -------
        list of dict
            The records produced, which are also in :attr:`history`.
        """
        return [self.process_chunk(X, y) for X, y in chunks]

    # --------------------------------------------------------------- output
    def get(self, field: str) -> List[Any]:
        """One column of the log, ready to plot.

        Parameters
        ----------
        field : str
            A key from :attr:`FIELDS`.

        Returns
        -------
        list
            The value of ``field`` for every completed chunk.

        Raises
        ------
        KeyError
            If ``field`` is not a logged field.

        Examples
        --------
        >>> trainer = StreamTrainer.__new__(StreamTrainer)   # doctest: +SKIP
        >>> trainer.get("accuracy")                          # doctest: +SKIP
        """
        if field not in _FIELDS:
            raise KeyError(f"unknown field {field!r}; available: {list(_FIELDS)}")
        return [rec[field] for rec in self.history]

    def summary(self) -> Dict[str, Any]:
        """Totals for the run as a whole.

        Returns
        -------
        dict
            Keys ``n_chunks``, ``n_samples_seen``, ``n_retrains``,
            ``cumulative_accuracy``, ``total_train_seconds``,
            ``total_retrain_seconds``, ``total_predict_seconds``,
            ``final_memory_bytes`` and the accumulated ``precision``,
            ``recall`` and ``f1``.
        """
        def total(field: str) -> float:
            return float(sum(v for v in self.get(field) if v is not None))

        acc = self.metrics.result()
        return {
            "n_chunks": self.n_chunks_,
            "n_samples_seen": self.n_samples_seen_,
            "n_retrains": self._retrain_count(),
            "cumulative_accuracy": acc["accuracy"],
            "precision": acc["precision"],
            "recall": acc["recall"],
            "f1": acc["f1"],
            "total_train_seconds": total("train_seconds"),
            "total_retrain_seconds": total("retrain_seconds"),
            "total_predict_seconds": total("predict_seconds"),
            "final_memory_bytes": _memory_bytes(self.model),
        }

    def to_csv(self, path) -> str:
        """Write :attr:`history` to a CSV file using the standard library.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file; it is overwritten.

        Returns
        -------
        str
            The path written, so it can be printed or chained.

        Raises
        ------
        ValueError
            If no chunk has been completed yet.

        Notes
        -----
        Uses :mod:`csv` from the standard library — no pandas. Time
        O(n_chunks), space O(1).
        """
        if not self.history:
            raise ValueError("nothing to write: no chunk has been completed yet")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_FIELDS))
            writer.writeheader()
            writer.writerows(self.history)
        return str(path)

    def __repr__(self) -> str:
        return (
            f"StreamTrainer(model={type(self.model).__name__}, chunks={self.n_chunks_}, "
            f"samples={self.n_samples_seen_}, retrains={self._retrain_count()})"
        )
