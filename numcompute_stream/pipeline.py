"""Incremental pipeline for streaming machine-learning workflows.

Intermediate transformers are initialised on the first training chunk and
then kept fixed. This preserves a consistent feature representation for
estimators that buffer samples before retraining.
"""

from __future__ import annotations

import numpy as np

from numcompute.pipeline import Pipeline
from numcompute.utils import NotFittedError


__all__ = ["StreamingPipeline"]


class StreamingPipeline(Pipeline):
    """A Pipeline that supports incremental training with partial_fit.

    Parameters
    ----------
    steps : list of (str, object)
        Named transformers followed by an incremental estimator.

    Notes
    -----
    Intermediate transformers are fitted on the first chunk only.
    Subsequent chunks use their existing transformations. The final
    estimator must implement partial_fit and predict.
    """

    def __init__(self, steps):
        super().__init__(steps)
        self._stream_fitted = False

        final_estimator = self.steps[-1][1]
        if not hasattr(final_estimator, "partial_fit"):
            raise TypeError(
                "The final pipeline step must implement partial_fit"
            )

    def _transform_chunk(self, X, y=None, fit=False):
        """Apply intermediate transformers to a chunk."""
        Xt = np.asarray(X)

        for _, transformer in self.steps[:-1]:
            if fit:
                if hasattr(transformer, "partial_fit"):
                    transformer.partial_fit(Xt, y)
                else:
                    transformer.fit(Xt, y)

            Xt = transformer.transform(Xt)

        return Xt

    def partial_fit(self, X, y):
        """Fit on the first chunk and incrementally train on every chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.
        y : array_like, shape (n_samples,)
            Target labels.

        Returns
        -------
        StreamingPipeline
            The fitted pipeline.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if X.shape[0] == 0:
            raise ValueError("Training chunk must not be empty")

        Xt = self._transform_chunk(
            X, y, fit=not self._stream_fitted
        )

        self.steps[-1][1].partial_fit(Xt, y)
        self._stream_fitted = True

        return self

    def predict(self, X):
        """Predict labels for a new chunk."""
        if not self._stream_fitted:
            raise NotFittedError("StreamingPipeline has not been fitted")

        Xt = self._transform_chunk(X)
        return self.steps[-1][1].predict(Xt)

    def predict_proba(self, X):
        """Predict class probabilities when supported by the estimator."""
        if not self._stream_fitted:
            raise NotFittedError("StreamingPipeline has not been fitted")

        estimator = self.steps[-1][1]

        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "The final estimator does not implement predict_proba"
            )

        Xt = self._transform_chunk(X)
        return estimator.predict_proba(Xt)

    def memory_bytes(self):
        """Return the total memory reported by pipeline steps."""
        sizes = [
            int(step.memory_bytes())
            for _, step in self.steps
            if hasattr(step, "memory_bytes")
        ]
        return sum(sizes)