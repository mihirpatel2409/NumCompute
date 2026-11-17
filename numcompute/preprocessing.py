"""Preprocessing transformers: scalers, imputer and one-hot encoder.

All follow ``fit(X, y=None) -> self``, ``transform(X)``, ``fit_transform(X)``.
Scalers and the imputer ignore NaN when learning statistics and keep NaN in
their output where the input had it (the imputer fills it).

Streaming
---------
:class:`StandardScaler`, :class:`Imputer` and :class:`OneHotEncoder` also
support ``partial_fit(X_chunk, y=None)``, which folds one chunk into the
learned state without revisiting earlier chunks. ``fit`` always starts from
scratch; ``partial_fit`` accumulates. Both leave ``transform`` unchanged, so a
transformer can be used in batch or streaming mode interchangeably. Unlike
``fit``, ``partial_fit`` accepts an empty chunk (zero rows) and treats it as a
no-op, because a stream may legitimately deliver one.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pipeline import Transformer
from .stats import RunningStats, StreamingStats
from .utils import check_2d, check_is_fitted

__all__ = ["StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder"]

_EPS = np.finfo(float).eps


def _check_fit_data(X) -> np.ndarray:
    """Validate training data: 2-D, non-empty, no infinities."""
    X = check_2d(X, "X", allow_empty=False)
    if np.isinf(X).any():
        raise ValueError("X contains infinity; clip or remove infinite values first")
    return X


def _check_transform_data(est, X, n_features: int) -> np.ndarray:
    X = check_2d(X, "X")
    if X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(est).__name__} was fitted "
            f"with {n_features}"
        )
    return X


class StandardScaler(Transformer):
    """Standardise features to zero mean and unit variance (z-score).

    Parameters
    ----------
    with_mean : bool, optional
        Subtract the column mean. Default True.
    with_std : bool, optional
        Divide by the column standard deviation (ddof=0). Default True.

    Attributes
    ----------
    mean_, var_, scale_ : numpy.ndarray, shape (n_features,)
        Column mean, variance and the divisor actually used. Constant columns
        get ``scale_ = 1`` so they map to zeros instead of dividing by zero.
        All-NaN columns get ``mean_ = 0`` and ``scale_ = 1``.
    n_features_in_ : int
        Number of columns seen in ``fit``.
    n_samples_seen_ : int
        Rows folded into the statistics (reset by ``fit``, accumulated by
        ``partial_fit``).

    Notes
    -----
    NaN entries are ignored when fitting and stay NaN when transforming.
    Both ``fit`` and ``partial_fit`` accumulate through
    :class:`numcompute.stats.RunningStats`, so a chunked fit and a batch fit
    give the same statistics to floating-point tolerance.
    Time O(n d), space O(d) beyond the output.

    Examples
    --------
    >>> StandardScaler().fit_transform([[1.0], [3.0]]).ravel().tolist()
    [-1.0, 1.0]
    >>> s = StandardScaler().partial_fit([[1.0], [3.0]]).partial_fit([[5.0]])
    >>> float(s.mean_[0]), s.n_samples_seen_
    (3.0, 3)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._running = RunningStats()

    def _finalise(self, n_features: int) -> "StandardScaler":
        """Derive ``mean_``/``var_``/``scale_`` from the running state.

        Shared by ``fit`` and ``partial_fit`` so both apply exactly the same
        rules: a column with no observations is centred at 0, and a constant
        column gets ``scale_ = 1`` (mapping to zeros) instead of dividing by
        something indistinguishable from zero.
        """
        mean, var = self._running.mean, self._running.var()
        mean = np.where(np.isnan(mean), 0.0, mean)
        var = np.where(np.isnan(var), 0.0, var)
        scale = np.sqrt(var)
        scale = np.where(scale <= 10 * _EPS * np.maximum(1.0, np.abs(mean)), 1.0, scale)
        self.mean_, self.var_, self.scale_ = mean, var, scale
        self.n_features_in_ = n_features
        return self

    def fit(self, X, y=None) -> "StandardScaler":
        """Compute column means and standard deviations, discarding earlier state.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data (NaN allowed).
        y : ignored

        Returns
        -------
        StandardScaler
            ``self``.

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, is empty, or contains infinity.
        """
        X = _check_fit_data(X)
        self._running = RunningStats().update(X)
        self.n_samples_seen_ = int(X.shape[0])
        return self._finalise(X.shape[1])

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Fold one chunk into the running mean and variance.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk of training data (NaN allowed, zero rows allowed).
        y : ignored

        Returns
        -------
        StandardScaler
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, contains infinity, or its feature count
            differs from earlier chunks.

        Notes
        -----
        Uses :class:`numcompute.stats.RunningStats`, i.e. Welford's update
        generalised to batches by Chan et al.'s parallel merge, so chunks of
        different sizes are merged without loss of precision and the result
        does not depend on how the stream was split. Time O(n d) per chunk,
        space O(d) of retained state.
        """
        X = check_2d(X, "X")
        if np.isinf(X).any():
            raise ValueError("X contains infinity; clip or remove infinite values first")
        self._running.update(X)
        self.n_samples_seen_ = getattr(self, "n_samples_seen_", 0) + int(X.shape[0])
        return self._finalise(X.shape[1])

    def transform(self, X) -> np.ndarray:
        """Standardise ``X`` with the learned statistics.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            New array; ``X`` is not modified.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If the number of features differs from ``fit``.
        """
        check_is_fitted(self, "scale_")
        X = _check_transform_data(self, X, self.n_features_in_)
        out = X - self.mean_ if self.with_mean else X.copy()
        if self.with_std:
            out = out / self.scale_
        return out

    def inverse_transform(self, X) -> np.ndarray:
        """Undo :meth:`transform`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)

        Raises
        ------
        NotFittedError, ValueError
            As for :meth:`transform`.
        """
        check_is_fitted(self, "scale_")
        X = _check_transform_data(self, X, self.n_features_in_)
        out = X * self.scale_ if self.with_std else X.copy()
        return out + self.mean_ if self.with_mean else out


class MinMaxScaler(Transformer):
    """Rescale each feature linearly to ``feature_range``.

    Parameters
    ----------
    feature_range : tuple (low, high), optional
        Target range with ``low < high``. Default ``(0, 1)``.
    clip : bool, optional
        Clip transformed values of unseen data to ``feature_range``.

    Attributes
    ----------
    data_min_, data_max_, data_range_, scale_, min_ : numpy.ndarray, shape (n_features,)
        ``X_out = X * scale_ + min_``. Constant columns get ``data_range_ = 1``
        and map to ``low``.
    n_features_in_ : int

    Raises
    ------
    ValueError
        At ``fit`` if ``low >= high``.

    Notes
    -----
    NaN is ignored when fitting and preserved when transforming.
    Time O(n d), space O(d).

    Examples
    --------
    >>> MinMaxScaler((-1, 1)).fit_transform([[0.0], [5.0], [10.0]]).ravel().tolist()
    [-1.0, 0.0, 1.0]
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), clip: bool = False):
        self.feature_range = feature_range
        self.clip = clip

    def fit(self, X, y=None) -> "MinMaxScaler":
        """Learn per-column minimum and maximum.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        MinMaxScaler
            ``self``.

        Raises
        ------
        TypeError, ValueError
            If ``X`` is invalid or ``feature_range`` is not ``low < high``.
        """
        try:
            low, high = self.feature_range
            low, high = float(low), float(high)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_range must be a (low, high) pair of numbers") from exc
        if not low < high:
            raise ValueError(f"feature_range must satisfy low < high, got {self.feature_range}")
        X = _check_fit_data(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dmin, dmax = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        dmin = np.where(np.isnan(dmin), 0.0, dmin)
        dmax = np.where(np.isnan(dmax), 0.0, dmax)
        drange = dmax - dmin
        drange = np.where(drange == 0.0, 1.0, drange)
        self.data_min_, self.data_max_, self.data_range_ = dmin, dmax, drange
        self.scale_ = (high - low) / drange
        self.min_ = low - dmin * self.scale_
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        """Scale ``X`` to the target range.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If the number of features differs from ``fit``.
        """
        check_is_fitted(self, "scale_")
        X = _check_transform_data(self, X, self.n_features_in_)
        out = X * self.scale_ + self.min_
        if self.clip:
            low, high = float(self.feature_range[0]), float(self.feature_range[1])
            out = np.clip(out, low, high)
        return out

    def inverse_transform(self, X) -> np.ndarray:
        """Undo :meth:`transform` (clipping cannot be undone).

        Raises
        ------
        NotFittedError, ValueError
            As for :meth:`transform`.
        """
        check_is_fitted(self, "scale_")
        X = _check_transform_data(self, X, self.n_features_in_)
        return (X - self.min_) / self.scale_


class Imputer(Transformer):
    """Replace missing values using column statistics or a constant.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, optional
        Replacement rule per column. ``most_frequent`` picks the smallest
        value among ties. Default ``'mean'``.
    fill_value : float, optional
        Used by ``'constant'`` and for columns that are entirely missing.
        Default 0.0.
    missing_values : float, optional
        What counts as missing: ``np.nan`` (default) or a sentinel such as -999.

    Attributes
    ----------
    statistics_ : numpy.ndarray, shape (n_features,)
        Replacement value of each column.
    n_features_in_ : int
    n_samples_seen_ : int
        Rows folded into the statistics.
    n_missing_ : numpy.ndarray, shape (n_features,)
        Missing entries counted per column across everything seen.

    Raises
    ------
    ValueError
        At ``fit`` for an unknown ``strategy``.

    Notes
    -----
    ``fit`` discards any state accumulated by ``partial_fit`` and learns from
    its argument alone; a later ``partial_fit`` therefore starts a new
    accumulation. See :meth:`partial_fit` for which strategies are exact in
    streaming mode.
    Time O(n d) for mean/median (most_frequent adds a sort per column,
    O(n log n d)); space O(n d) for the missing-value mask.

    Examples
    --------
    >>> Imputer("mean").fit_transform([[1.0], [np.nan], [3.0]]).ravel().tolist()
    [1.0, 2.0, 3.0]
    >>> imp = Imputer("mean").partial_fit([[1.0], [np.nan]]).partial_fit([[3.0]])
    >>> float(imp.transform([[np.nan]])[0, 0]), imp.n_missing_.tolist()
    (2.0, [1.0])
    """

    _STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0, missing_values: float = np.nan):
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self._stream = None

    def _mask(self, X: np.ndarray) -> np.ndarray:
        mv = self.missing_values
        return np.isnan(X) if (mv is None or np.isnan(mv)) else (X == mv)

    def fit(self, X, y=None) -> "Imputer":
        """Learn the replacement value of each column.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        Imputer
            ``self``.

        Raises
        ------
        TypeError, ValueError
            If ``X`` is invalid or ``strategy`` is unknown.
        """
        if self.strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {self.strategy!r}")
        X = check_2d(X, "X", allow_empty=False)
        mask = self._mask(X)
        Xm = np.where(mask, np.nan, X)
        fill = float(self.fill_value)
        if self.strategy == "constant":
            stats = np.full(X.shape[1], fill)
        elif self.strategy in ("mean", "median"):
            fn = np.nanmean if self.strategy == "mean" else np.nanmedian
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats = fn(Xm, axis=0)
            stats = np.where(np.isnan(stats), fill, stats)
        else:
            stats = np.full(X.shape[1], fill)
            for j in range(X.shape[1]):  # per-column mode: loop over features only
                col = X[~mask[:, j], j]
                col = col[~np.isnan(col)]
                if col.size:
                    vals, counts = np.unique(col, return_counts=True)
                    stats[j] = vals[np.argmax(counts)]
        self.statistics_ = stats
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = int(X.shape[0])
        self.n_missing_ = mask.sum(axis=0).astype(float)
        self._stream = None
        return self

    def partial_fit(self, X, y=None) -> "Imputer":
        """Fold one chunk into the running replacement values.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk of training data (zero rows allowed).
        y : ignored

        Returns
        -------
        Imputer
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, its feature count differs from earlier
            chunks, or ``strategy='most_frequent'``.

        Notes
        -----
        Accuracy depends on the strategy:

        * ``'mean'`` and ``'constant'`` are **exact** — identical to what
          ``fit`` would produce on the concatenated stream.
        * ``'median'`` is **approximate**. An exact streaming median would
          have to retain every value; the estimate comes from the binned
          quantile of :class:`numcompute.stats.StreamingStats`, so its error
          is bounded by one histogram bin width.
        * ``'most_frequent'`` is rejected, because tracking the mode of an
          unbounded stream needs a count per distinct value. Use ``fit``.

        A column that is missing in every chunk seen so far keeps
        ``fill_value`` as its replacement, exactly as in ``fit``.
        :attr:`n_missing_` tracks how many entries were missing per column.
        Time O(n d) per chunk, space O(d * bins) of retained state.
        """
        if self.strategy not in ("mean", "median", "constant"):
            raise ValueError(
                f"partial_fit supports strategy 'mean', 'median' and 'constant', got "
                f"{self.strategy!r}; 'most_frequent' needs the whole dataset, use fit instead"
            )
        X = check_2d(X, "X")
        mask = self._mask(X)
        fill = float(self.fill_value)
        if getattr(self, "_stream", None) is None:
            self._stream = StreamingStats()
            self.n_samples_seen_ = 0
            self.n_missing_ = np.zeros(X.shape[1])
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but earlier chunks had {self.n_features_in_}"
            )
        self._stream.update_stats(np.where(mask, np.nan, X))
        self.n_samples_seen_ += int(X.shape[0])
        self.n_missing_ += mask.sum(axis=0)
        self.n_features_in_ = X.shape[1]
        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], fill)
        elif self.strategy == "mean":
            self.statistics_ = np.where(np.isnan(self._stream.mean), fill, self._stream.mean)
        elif self._stream.has_data:
            med = self._stream.quantile(0.5)
            self.statistics_ = np.where(np.isnan(med), fill, med)
        else:  # every value seen so far was missing
            self.statistics_ = np.full(X.shape[1], fill)
        return self

    def transform(self, X) -> np.ndarray:
        """Fill missing entries of ``X``.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            New array without the missing entries.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If the number of features differs from ``fit``.
        """
        check_is_fitted(self, "statistics_")
        X = _check_transform_data(self, X, self.n_features_in_)
        return np.where(self._mask(X), self.statistics_, X)


class OneHotEncoder(Transformer):
    """Encode categorical columns as binary indicator columns.

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, optional
        Behaviour for categories not seen in ``fit``: raise, or encode as an
        all-zero block. Default ``'error'``.
    dtype : type, optional
        Output dtype. Default ``float``.

    Attributes
    ----------
    categories_ : list of numpy.ndarray
        Categories of each input column, **in output-column order**. After
        ``fit`` this is sorted; after ``partial_fit`` newly discovered
        categories are appended (see Notes).
    n_features_in_ : int
    n_output_features_ : int
        Total number of output columns.
    n_samples_seen_ : int
        Rows folded into ``categories_``.

    Notes
    -----
    Accepts string, integer or float data; a 1-D input is treated as one
    column. Float data containing NaN is rejected (impute first). Output has
    one block per input column, laid out in input order. Time O(n d log c)
    (binary search per column), space O(n * total_categories).

    **Growing categories.** ``partial_fit`` appends categories it has not seen
    before to the end of that column's block rather than re-sorting. This is
    deliberate: the column index of every previously known category is
    therefore permanent, so a 1 in output column *k* means the same thing in
    every chunk, before and after the category set grows. The only effect of
    growth is that later chunks produce *wider* output. Encoded data from an
    earlier chunk stays valid as a prefix of the current layout, so aligning
    it only requires padding zero columns on the right — zeros being the
    correct encoding, since those categories genuinely did not occur. Use
    :attr:`n_output_features_` to detect that a widening has happened.

    Examples
    --------
    >>> OneHotEncoder().fit_transform([["a"], ["b"], ["a"]]).tolist()
    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    >>> enc = OneHotEncoder().partial_fit([["b"], ["a"]]).partial_fit([["c"]])
    >>> enc.categories_[0].tolist()          # 'c' appended, 'a'/'b' keep their columns
    ['a', 'b', 'c']
    >>> enc.transform([["b"]]).tolist()
    [[0.0, 1.0, 0.0]]
    """

    def __init__(self, handle_unknown: str = "error", dtype=float):
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {X.shape}")
        if X.dtype.kind == "f" and np.isnan(X).any():
            raise ValueError("X contains NaN; impute it first or encode it as a category string")
        return X

    def fit(self, X, y=None) -> "OneHotEncoder":
        """Learn the categories of every column.

        Parameters
        ----------
        X : array_like, shape (n_samples,) or (n_samples, n_features)
        y : ignored

        Returns
        -------
        OneHotEncoder
            ``self``.

        Raises
        ------
        ValueError
            If ``X`` is empty, has more than 2 dimensions, contains NaN, or
            ``handle_unknown`` is invalid.
        """
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {self.handle_unknown!r}")
        X = self._as_2d(X)
        if X.size == 0:
            raise ValueError("X must not be empty")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = int(X.shape[0])
        return self._refresh_layout()

    def _refresh_layout(self) -> "OneHotEncoder":
        """Recompute the output width and the sorted lookup table.

        ``categories_`` is not guaranteed to be sorted once ``partial_fit``
        has appended to it, so membership tests use a separately sorted copy
        and map back to the real output column.
        """
        self.n_output_features_ = int(sum(len(c) for c in self.categories_))
        self._lookup = []
        for cats in self.categories_:
            order = np.argsort(cats, kind="stable")
            self._lookup.append((cats[order], order))
        return self

    def partial_fit(self, X, y=None) -> "OneHotEncoder":
        """Fold one chunk into the known categories, appending new ones.

        Parameters
        ----------
        X : array_like, shape (n_samples,) or (n_samples, n_features)
            Chunk of training data (zero rows allowed).
        y : ignored

        Returns
        -------
        OneHotEncoder
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If ``X`` has more than 2 dimensions, contains NaN,
            ``handle_unknown`` is invalid, or the feature count differs from
            earlier chunks.

        Notes
        -----
        Categories unseen so far are appended in sorted order to the end of
        their column's block, so existing output columns never change
        meaning; see the class Notes for what that implies for data encoded
        before the growth. Time O(n d log c) per chunk, space O(total
        categories).
        """
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {self.handle_unknown!r}")
        X = self._as_2d(X)
        if not hasattr(self, "categories_"):
            self.categories_ = [np.empty(0, dtype=X.dtype) for _ in range(X.shape[1])]
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = 0
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but earlier chunks had {self.n_features_in_}"
            )
        for j in range(X.shape[1]):
            fresh = np.setdiff1d(np.unique(X[:, j]), self.categories_[j])
            if fresh.size:
                self.categories_[j] = np.concatenate([self.categories_[j], fresh])
        self.n_samples_seen_ += int(X.shape[0])
        return self._refresh_layout()

    def transform(self, X) -> np.ndarray:
        """Encode ``X``.

        Parameters
        ----------
        X : array_like, shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_output_features_)
            Indicator matrix.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If the feature count differs from ``fit``, or an unseen category
            appears while ``handle_unknown='error'``.
        """
        check_is_fitted(self, "categories_")
        X = self._as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoder was fitted with "
                f"{self.n_features_in_}"
            )
        n = X.shape[0]
        blocks: List[np.ndarray] = []
        for j, (sorted_cats, order) in enumerate(self._lookup):  # loop over columns, not rows
            col = X[:, j]
            n_cat = sorted_cats.size
            block = np.zeros((n, n_cat), dtype=self.dtype)
            if n_cat:
                pos = np.minimum(np.searchsorted(sorted_cats, col), n_cat - 1)
                known = sorted_cats[pos] == col
            else:  # a column whose categories are still unknown encodes to nothing
                known = np.zeros(n, dtype=bool)
            if self.handle_unknown == "error" and not known.all():
                bad = np.unique(col[~known])[:5].tolist()
                raise ValueError(f"unknown categories {bad} in column {j}")
            if n_cat:
                rows = np.nonzero(known)[0]
                block[rows, order[pos[rows]]] = 1  # order maps sorted position -> output column
            blocks.append(block)
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        """Names of the output columns, e.g. ``'color_red'``.

        Parameters
        ----------
        input_features : sequence of str, optional
            Input column names; defaults to ``x0, x1, ...``.

        Returns
        -------
        list of str
            One name per output column.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If ``input_features`` has the wrong length.
        """
        check_is_fitted(self, "categories_")
        names = list(input_features) if input_features is not None else [f"x{j}" for j in range(self.n_features_in_)]
        if len(names) != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} feature names, got {len(names)}")
        return [f"{nm}_{c}" for nm, cats in zip(names, self.categories_) for c in cats]

    def inverse_transform(self, Z) -> np.ndarray:
        """Convert indicator columns back to categories.

        Parameters
        ----------
        Z : array_like, shape (n_samples, n_output_features_)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_features)
            Recovered categories. Rows that are all zero (unknown category,
            only possible with ``handle_unknown='ignore'``) give ``None`` and
            the result has dtype object.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If ``Z`` has the wrong width.
        """
        check_is_fitted(self, "categories_")
        Z = check_2d(Z, "Z")
        if Z.shape[1] != self.n_output_features_:
            raise ValueError(f"Z must have {self.n_output_features_} columns, got {Z.shape[1]}")
        cols, start, unknown = [], 0, False
        for cats in self.categories_:
            block = Z[:, start:start + len(cats)]
            start += len(cats)
            picked = cats[np.argmax(block, axis=1)]
            empty = block.max(axis=1) == 0
            if empty.any():
                unknown = True
                picked = picked.astype(object)
                picked[empty] = None
            cols.append(picked)
        out = np.column_stack(cols)
        return out.astype(object) if unknown else out
