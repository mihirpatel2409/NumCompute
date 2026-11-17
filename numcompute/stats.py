"""Descriptive statistics, streaming (Welford/Chan) statistics, histogram, quantiles.

NaN policy (``nan_policy`` argument)
------------------------------------
``'propagate'`` (default) - a NaN in a slice makes that slice's result NaN.
``'omit'``                - NaNs are ignored; an all-NaN slice gives NaN.
``'raise'``               - ``ValueError`` if any NaN is present.
``histogram`` accepts only ``'omit'`` (default) and ``'raise'``.

Axis semantics: ``axis=None`` reduces over all elements and returns a scalar;
an integer axis removes that axis from the output shape.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .rank import percentile
from .utils import as_float_array, check_nan_policy, normalize_axis

__all__ = [
    "mean",
    "median",
    "var",
    "std",
    "minimum",
    "maximum",
    "describe",
    "RunningStats",
    "StreamingStats",
    "histogram",
    "quantile",
]

Axis = Union[int, None]


def _prepare(a, axis: Axis, nan_policy: str) -> np.ndarray:
    check_nan_policy(nan_policy)
    arr = as_float_array(a, name="a", allow_empty=False)
    if axis is not None:
        if arr.ndim == 0:
            raise ValueError("cannot use axis with a 0-d array")
        normalize_axis(axis, arr.ndim)
    if nan_policy == "raise" and np.isnan(arr).any():
        raise ValueError("a contains NaN")
    return arr


def mean(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Arithmetic mean.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    axis : int or None, optional
        Axis to reduce; ``None`` for all elements.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    float or numpy.ndarray
        Scalar if ``axis`` is None, otherwise ``a.shape`` without ``axis``.

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``a`` is empty, ``axis``/``nan_policy`` is invalid, or NaN is
        present with ``nan_policy='raise'``.

    Notes
    -----
    Time O(n), space O(1) extra.

    Examples
    --------
    >>> float(mean([1, 2, 3, 4]))
    2.5
    """
    arr = _prepare(a, axis, nan_policy)
    if nan_policy == "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(arr, axis=axis)
    return np.mean(arr, axis=axis)


def median(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Median (50th percentile).

    Parameters
    ----------
    a, axis, nan_policy
        As in :func:`mean`.

    Returns
    -------
    float or numpy.ndarray
        Same shape rules as :func:`mean`.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Implemented with :func:`numcompute.rank.percentile`. Time O(n log n),
    space O(n).

    Examples
    --------
    >>> float(median([3, 1, 2, 10]))
    2.5
    """
    arr = _prepare(a, axis, nan_policy)
    return percentile(arr, 50, axis=axis, nan_policy=nan_policy)


def var(a, axis: Axis = None, ddof: int = 0, nan_policy: str = "propagate"):
    """Variance.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    axis : int or None, optional
        Axis to reduce.
    ddof : int, optional
        Delta degrees of freedom: divisor is ``n - ddof``. Default 0.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    float or numpy.ndarray
        NaN where ``n - ddof <= 0``.

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``ddof < 0`` or an argument is invalid, or the input is empty.

    Notes
    -----
    Two-pass algorithm (mean first), which is numerically stable. Time O(n),
    space O(n).

    Examples
    --------
    >>> float(var([1, 2, 3, 4], ddof=1))
    1.6666666666666667
    """
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError(f"ddof must be a non-negative int, got {ddof!r}")
    arr = _prepare(a, axis, nan_policy)
    fn = np.nanvar if nan_policy == "omit" else np.var
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            out = fn(arr, axis=axis, ddof=ddof)
    count = np.sum(~np.isnan(arr), axis=axis) if nan_policy == "omit" else (
        arr.size if axis is None else arr.shape[axis]
    )
    out = np.where(np.asarray(count) - ddof <= 0, np.nan, out)
    return out[()] if out.ndim == 0 else out


def std(a, axis: Axis = None, ddof: int = 0, nan_policy: str = "propagate"):
    """Standard deviation (square root of :func:`var`).

    Parameters
    ----------
    a, axis, ddof, nan_policy
        As in :func:`var`.

    Returns
    -------
    float or numpy.ndarray
        Same shape rules as :func:`var`.

    Raises
    ------
    TypeError, ValueError
        As in :func:`var`.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> float(std([2, 4, 4, 4, 5, 5, 7, 9]))
    2.0
    """
    return np.sqrt(var(a, axis=axis, ddof=ddof, nan_policy=nan_policy))


def minimum(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Smallest value.

    Parameters
    ----------
    a, axis, nan_policy
        As in :func:`mean`.

    Returns
    -------
    float or numpy.ndarray
        All-NaN slices give NaN under ``'omit'``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Time O(n), space O(1) extra.
    """
    arr = _prepare(a, axis, nan_policy)
    if nan_policy == "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmin(arr, axis=axis)
    return np.min(arr, axis=axis)


def maximum(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Largest value.

    Parameters
    ----------
    a, axis, nan_policy
        As in :func:`mean`.

    Returns
    -------
    float or numpy.ndarray
        All-NaN slices give NaN under ``'omit'``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Time O(n), space O(1) extra.
    """
    arr = _prepare(a, axis, nan_policy)
    if nan_policy == "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(arr, axis=axis)
    return np.max(arr, axis=axis)


def describe(a, axis: Axis = None, nan_policy: str = "propagate") -> Dict[str, object]:
    """Summary statistics in one call.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    axis : int or None, optional
        Axis to summarise.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    dict
        Keys ``count`` (non-NaN values), ``mean``, ``std`` (ddof=0), ``min``,
        ``q25``, ``median``, ``q75``, ``max``. Values are scalars or arrays
        shaped like the reduced output.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Time O(n log n), space O(n).
    """
    arr = _prepare(a, axis, nan_policy)
    q25, med, q75 = percentile(arr, [25, 50, 75], axis=axis, nan_policy=nan_policy)
    count = np.sum(~np.isnan(arr), axis=axis)
    return {
        "count": count,
        "mean": mean(arr, axis, nan_policy),
        "std": std(arr, axis, 0, nan_policy),
        "min": minimum(arr, axis, nan_policy),
        "q25": q25,
        "median": med,
        "q75": q75,
        "max": maximum(arr, axis, nan_policy),
    }


class RunningStats:
    """Streaming per-feature count, mean and variance (Welford / Chan).

    Feed data in batches with :meth:`update`; memory stays O(n_features) no
    matter how many rows are seen. Batches are merged with Chan et al.'s
    parallel formula, which is numerically stable and equals Welford's
    update when a batch has one row. NaN entries are skipped per feature.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_features_ : int or None
        Number of features (set by the first update).
    count : numpy.ndarray, shape (n_features,)
        Non-NaN observations per feature.
    mean : numpy.ndarray, shape (n_features,)
        Running mean (NaN for features with no observations).

    Examples
    --------
    >>> rs = RunningStats().update([[1.0], [2.0]]).update([[3.0], [4.0]])
    >>> float(rs.mean[0]), float(rs.var(ddof=1)[0])
    (2.5, 1.6666666666666667)
    """

    def __init__(self) -> None:
        self.n_features_: Optional[int] = None
        self._n = np.zeros(0)
        self._mean = np.zeros(0)
        self._m2 = np.zeros(0)

    def _init(self, n_features: int) -> None:
        self.n_features_ = n_features
        self._n = np.zeros(n_features)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)

    def _merge(self, n_b, mean_b, m2_b) -> None:
        total = self._n + n_b
        delta = mean_b - self._mean
        frac = n_b / np.maximum(total, 1.0)
        self._mean = self._mean + delta * frac
        self._m2 = self._m2 + m2_b + delta**2 * self._n * frac
        self._n = total

    def update(self, x) -> "RunningStats":
        """Add a batch of observations.

        Parameters
        ----------
        x : array_like, shape (n_rows, n_features) or (n_rows,)
            New data. A 1-D input is treated as one feature.

        Returns
        -------
        RunningStats
            ``self`` (so calls can be chained).

        Raises
        ------
        TypeError
            If ``x`` is not numeric.
        ValueError
            If ``x`` is not 1-D/2-D or its feature count differs from earlier batches.

        Notes
        -----
        Time O(n_rows * n_features), space O(n_rows * n_features) for the
        batch (state is O(n_features)).
        """
        arr = as_float_array(x, name="x", ndim=(1, 2))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.n_features_ is None:
            self._init(arr.shape[1])
        elif arr.shape[1] != self.n_features_:
            raise ValueError(
                f"x has {arr.shape[1]} features, but earlier batches had {self.n_features_}"
            )
        if arr.shape[0] == 0:
            return self
        valid = ~np.isnan(arr)
        n_b = valid.sum(axis=0).astype(float)
        filled = np.where(valid, arr, 0.0)
        mean_b = filled.sum(axis=0) / np.maximum(n_b, 1.0)
        m2_b = (((filled - mean_b) ** 2) * valid).sum(axis=0)
        self._merge(n_b, mean_b, m2_b)
        return self

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Combine another ``RunningStats`` into this one.

        Parameters
        ----------
        other : RunningStats
            Statistics of another stream with the same number of features.

        Returns
        -------
        RunningStats
            ``self``.

        Raises
        ------
        TypeError
            If ``other`` is not a ``RunningStats``.
        ValueError
            If the feature counts differ.

        Notes
        -----
        Time O(n_features), space O(n_features).
        """
        if not isinstance(other, RunningStats):
            raise TypeError("other must be a RunningStats")
        if other.n_features_ is None:
            return self
        if self.n_features_ is None:
            self._init(other.n_features_)
        elif self.n_features_ != other.n_features_:
            raise ValueError(
                f"feature counts differ: {self.n_features_} and {other.n_features_}"
            )
        self._merge(other._n, other._mean, other._m2)
        return self

    @property
    def count(self) -> np.ndarray:
        """Non-NaN observation count per feature."""
        return self._n.copy()

    @property
    def mean(self) -> np.ndarray:
        """Running mean per feature (NaN where nothing was observed)."""
        return np.where(self._n > 0, self._mean, np.nan)

    def var(self, ddof: int = 0) -> np.ndarray:
        """Running variance per feature.

        Parameters
        ----------
        ddof : int, optional
            Divisor is ``count - ddof``. Default 0.

        Returns
        -------
        numpy.ndarray, shape (n_features,)
            NaN where ``count - ddof <= 0``.

        Raises
        ------
        ValueError
            If ``ddof < 0``.
        """
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")
        denom = self._n - ddof
        return np.where(denom > 0, self._m2 / np.where(denom > 0, denom, 1.0), np.nan)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Running standard deviation (square root of :meth:`var`)."""
        return np.sqrt(self.var(ddof))


def histogram(
    a,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    nan_policy: str = "omit",
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of a flattened array.

    Parameters
    ----------
    a : array_like of numbers
        Input values (flattened).
    bins : int or array_like, optional
        Number of equal-width bins, or strictly increasing bin edges
        (``len(bins) - 1`` bins). Default 10.
    range : (float, float), optional
        Lower and upper limit for integer ``bins``. Default: data min/max.
        If min equals max, the range is widened by 0.5 on each side.
    density : bool, optional
        Return probability density (integrates to 1 over the bins) instead
        of counts.
    nan_policy : {'omit', 'raise'}, optional
        NaN handling; ``'propagate'`` is not meaningful here.

    Returns
    -------
    counts : numpy.ndarray, shape (n_bins,)
        Integer counts (float if ``density``). Bins are half-open ``[l, r)``
        except the last, which includes its right edge.
    edges : numpy.ndarray, shape (n_bins + 1,)

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``bins`` is not positive/increasing, ``range`` is invalid, the
        automatic range is not finite, or NaN is present with ``'raise'``.

    Notes
    -----
    Values outside the edges are ignored. One ``searchsorted`` plus one
    ``bincount``: time O(n log b), space O(n).

    Examples
    --------
    >>> counts, edges = histogram([0, 1, 1, 2, 4], bins=2)
    >>> counts.tolist(), edges.tolist()
    ([3, 2], [0.0, 2.0, 4.0])
    """
    check_nan_policy(nan_policy, allowed=("omit", "raise"))
    arr = as_float_array(a, name="a").ravel()
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        if nan_policy == "raise":
            raise ValueError("a contains NaN")
        arr = arr[~nan_mask]

    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
            if not lo <= hi:
                raise ValueError(f"range must satisfy low <= high, got {range}")
        elif arr.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(arr.min()), float(arr.max())
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("data range is not finite; pass range=(low, high) explicitly")
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        edges = as_float_array(bins, name="bins", ndim=1)
        if edges.size < 2 or np.isnan(edges).any() or not (np.diff(edges) > 0).all():
            raise ValueError("bin edges must be a strictly increasing 1-D array of length >= 2")
    n_bins = edges.size - 1
    idx = np.searchsorted(edges, arr, side="right") - 1
    idx[arr == edges[-1]] = n_bins - 1
    inside = (idx >= 0) & (idx < n_bins)
    counts = np.bincount(idx[inside], minlength=n_bins)
    if density:
        total = counts.sum()
        widths = np.diff(edges)
        counts = counts / (total * widths) if total > 0 else np.zeros(n_bins)
    return counts, edges


def quantile(
    a,
    q,
    axis: Axis = None,
    interpolation: str = "linear",
    nan_policy: str = "propagate",
):
    """Quantiles (``q`` in [0, 1]) with NaN handling.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    q : float or array_like of float, 1-D
        Quantile(s) in ``[0, 1]``.
    axis : int or None, optional
        Axis to reduce; ``None`` flattens.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional
        As in :func:`numcompute.rank.percentile`.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    float or numpy.ndarray
        Shape ``q.shape + reduced_shape``; scalar for scalar ``q`` and
        ``axis=None``.

    Raises
    ------
    TypeError
        If ``a`` or ``q`` is not numeric.
    ValueError
        If ``q`` is outside [0, 1] or any other argument is invalid.

    Notes
    -----
    Thin wrapper over :func:`numcompute.rank.percentile`. Time O(n log n),
    space O(n).

    Examples
    --------
    >>> quantile([1, 2, 3, 4, 5], [0.25, 0.5]).tolist()
    [2.0, 3.0]
    """
    q_arr = as_float_array(q, name="q")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 1).any():
        raise ValueError("q must lie in [0, 1]")
    return percentile(a, q_arr * 100.0, axis=axis, interpolation=interpolation, nan_policy=nan_policy)


class StreamingStats:
    """Per-feature statistics maintained incrementally over a stream of chunks.

    Feed chunks with :meth:`update_stats`; the state is O(n_features * bins)
    regardless of how many rows pass through.

    Accuracy of each statistic
    --------------------------
    ``count``, ``mean``, ``var``, ``std``, ``minimum`` and ``maximum`` are
    **exact**: mean and variance delegate to :class:`RunningStats` (Welford's
    update generalised to batches by Chan et al.'s parallel merge), so the
    result equals a batch computation to floating-point tolerance however the
    data are chunked.

    ``quantile`` and ``histogram`` are **approximate**. An exact streaming
    quantile would have to retain every value seen; this class instead keeps a
    fixed grid of ``bins`` equal-width counters per feature. Within the binned
    range the quantile error is therefore bounded by one bin width,
    ``(hi - lo) / bins``. Values falling outside the grid are counted in an
    underflow/overflow counter and, because the exact running minimum and
    maximum are also kept, quantiles stay inside ``[minimum, maximum]``. Pass
    ``value_range`` when the data range is known in advance to make the bound
    tight from the first chunk; otherwise the grid is sized from the first
    chunk that contains data.

    Parameters
    ----------
    bins : int, optional
        Number of equal-width histogram bins per feature, ``>= 1``. Default 64.
    value_range : (float, float), optional
        Fixed ``(low, high)`` grid shared by every feature. Default ``None``,
        which sizes each feature's grid from the first non-empty chunk.
    nan_policy : {'omit', 'raise'}, optional
        ``'omit'`` (default) ignores NaN per feature; ``'raise'`` rejects any
        chunk containing NaN. ``'propagate'`` is not meaningful for a stream.

    Attributes
    ----------
    n_features_ : int or None
        Set by the first update.
    n_samples_seen_ : int
        Rows passed to :meth:`update_stats`, including rows that were all NaN.
    underflow_, overflow_ : numpy.ndarray, shape (n_features,)
        Non-NaN values that fell below / above the binned range.

    Raises
    ------
    ValueError
        If ``bins < 1``, ``value_range`` is not ``low <= high``, or
        ``nan_policy`` is not ``'omit'``/``'raise'``.

    Notes
    -----
    Time O(n_rows * n_features) per chunk, space O(n_features * bins).

    Examples
    --------
    >>> s = StreamingStats().update_stats([[1.0], [2.0]]).update_stats([[3.0], [4.0]])
    >>> float(s.mean[0]), float(s.var(ddof=1)[0]), int(s.count[0])
    (2.5, 1.6666666666666667, 4)
    """

    def __init__(
        self,
        bins: int = 64,
        value_range: Optional[Tuple[float, float]] = None,
        nan_policy: str = "omit",
    ) -> None:
        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(f"bins must be an int >= 1, got {bins!r}")
        check_nan_policy(nan_policy, allowed=("omit", "raise"))
        if value_range is not None:
            lo, hi = float(value_range[0]), float(value_range[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or not lo <= hi:
                raise ValueError(
                    f"value_range must be a finite (low, high) pair with low <= high, got {value_range}"
                )
            value_range = (lo, hi)
        self.bins = int(bins)
        self.value_range = value_range
        self.nan_policy = nan_policy
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> "StreamingStats":
        """Forget everything seen so far.

        Returns
        -------
        StreamingStats
            ``self``, so calls can be chained.

        Notes
        -----
        Time O(1), space O(1).
        """
        self.n_features_: Optional[int] = None
        self.n_samples_seen_ = 0
        self._rs = RunningStats()
        self._counts = np.zeros((0, self.bins))
        self._edges = np.zeros((0, self.bins + 1))
        self.underflow_ = np.zeros(0)
        self.overflow_ = np.zeros(0)
        self._min = np.zeros(0)
        self._max = np.zeros(0)
        self._edge_ready = np.zeros(0, dtype=bool)
        return self

    def _init(self, n_features: int) -> None:
        self.n_features_ = n_features
        self._counts = np.zeros((n_features, self.bins))
        self._edges = np.zeros((n_features, self.bins + 1))
        self.underflow_ = np.zeros(n_features)
        self.overflow_ = np.zeros(n_features)
        self._min = np.full(n_features, np.nan)
        self._max = np.full(n_features, np.nan)
        self._edge_ready = np.zeros(n_features, dtype=bool)
        if self.value_range is not None:
            lo, hi = self.value_range
            self._set_edges(np.ones(n_features, dtype=bool),
                            np.full(n_features, lo), np.full(n_features, hi))

    def _set_edges(self, which: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        """Fix the uniform grid of the selected features; flat ranges are widened.

        A feature's grid is built once, from the first chunk in which that
        feature has at least one non-NaN value (or from ``value_range``), and
        never moves afterwards, so counters stay comparable across chunks.
        """
        lo = np.asarray(lo, dtype=float).copy()
        hi = np.asarray(hi, dtype=float).copy()
        flat = which & ~(hi > lo)  # constant feature: widen by 0.5 on each side
        lo[flat] -= 0.5
        hi[flat] += 0.5
        step = (hi - lo) / self.bins
        grid = lo[:, None] + step[:, None] * np.arange(self.bins + 1)[None, :]
        self._edges[which] = grid[which]
        self._edge_ready |= which

    @property
    def _has_edges(self) -> bool:
        return bool(self._edge_ready.any())

    @property
    def has_data(self) -> bool:
        """True once at least one non-NaN value has been seen."""
        return self._has_edges

    # ----------------------------------------------------------------- update
    def update_stats(self, X_chunk) -> "StreamingStats":
        """Fold one chunk into the running statistics.

        Parameters
        ----------
        X_chunk : array_like, shape (n_rows, n_features) or (n_rows,)
            New rows. A 1-D chunk is treated as one feature. An empty chunk
            (zero rows) is accepted and changes nothing.

        Returns
        -------
        StreamingStats
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If ``X_chunk`` is not numeric.
        ValueError
            If ``X_chunk`` is not 1-D/2-D, its feature count differs from
            earlier chunks, it contains an infinity, or it contains NaN while
            ``nan_policy='raise'``.

        Notes
        -----
        Binning is vectorised: because the grid is uniform the bin index is an
        arithmetic expression, and the counters are accumulated with a single
        ``numpy.bincount``. Time O(n_rows * n_features), space O(n_rows *
        n_features) for the chunk.
        """
        arr = as_float_array(X_chunk, name="X_chunk", ndim=(1, 2))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.n_features_ is None:
            self._init(arr.shape[1])
        elif arr.shape[1] != self.n_features_:
            raise ValueError(
                f"X_chunk has {arr.shape[1]} features, but earlier chunks had {self.n_features_}"
            )
        if arr.shape[0] == 0:
            return self
        nan_mask = np.isnan(arr)
        if nan_mask.any() and self.nan_policy == "raise":
            raise ValueError("X_chunk contains NaN")
        if np.isinf(arr).any():
            raise ValueError(
                "X_chunk contains an infinity; clip or remove infinite values before "
                "streaming them (they would destroy the running mean and the histogram grid)"
            )

        self.n_samples_seen_ += int(arr.shape[0])
        self._rs.update(arr)
        valid = ~nan_mask
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chunk_min, chunk_max = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
        self._min = np.fmin(self._min, chunk_min)
        self._max = np.fmax(self._max, chunk_max)
        new_grid = ~self._edge_ready & ~np.isnan(self._min)
        if new_grid.any():
            self._set_edges(new_grid, self._min, self._max)
        self._bin(arr, valid & self._edge_ready)
        return self

    def _bin(self, arr: np.ndarray, valid: np.ndarray) -> None:
        """Accumulate histogram counters for one chunk (fully vectorised)."""
        d, b = self.n_features_, self.bins
        lo, hi = self._edges[:, 0], self._edges[:, -1]
        width = np.where(hi > lo, (hi - lo) / b, 1.0)
        safe = np.where(valid, arr, lo)  # NaN placeholder, masked out below
        idx = np.floor((safe - lo) / width).astype(np.intp)
        idx = np.where(safe == hi, b - 1, idx)  # last bin includes its right edge
        under = (idx < 0) & valid
        over = (idx >= b) & valid
        inside = valid & ~under & ~over
        self.underflow_ += under.sum(axis=0)
        self.overflow_ += over.sum(axis=0)
        if inside.any():
            cols = np.broadcast_to(np.arange(d), arr.shape)
            flat = cols[inside] * b + idx[inside]
            self._counts += np.bincount(flat, minlength=d * b).reshape(d, b)

    # ------------------------------------------------------------- statistics
    @property
    def count(self) -> np.ndarray:
        """Non-NaN observations per feature, shape ``(n_features,)``."""
        return self._rs.count

    @property
    def mean(self) -> np.ndarray:
        """Exact running mean per feature (NaN where nothing was observed)."""
        return self._rs.mean

    def var(self, ddof: int = 0) -> np.ndarray:
        """Exact running variance per feature; see :meth:`RunningStats.var`."""
        return self._rs.var(ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Exact running standard deviation per feature."""
        return self._rs.std(ddof)

    def minimum(self) -> np.ndarray:
        """Exact running minimum per feature (NaN where nothing was observed)."""
        return self._min.copy() if self.n_features_ is not None else np.zeros(0)

    def maximum(self) -> np.ndarray:
        """Exact running maximum per feature (NaN where nothing was observed)."""
        return self._max.copy() if self.n_features_ is not None else np.zeros(0)

    def histogram(self, feature: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Counts and edges of one feature's binned range.

        Parameters
        ----------
        feature : int, optional
            Feature index (negative indices allowed). Default 0.

        Returns
        -------
        counts : numpy.ndarray, shape (bins,)
            Values that fell inside the grid. Values outside it are in
            :attr:`underflow_` / :attr:`overflow_` instead.
        edges : numpy.ndarray, shape (bins + 1,)
            Uniform bin edges; the last bin includes its right edge.

        Raises
        ------
        ValueError
            If no chunk has been seen yet or ``feature`` is out of range.

        Notes
        -----
        Time O(bins), space O(bins).
        """
        self._require_data("histogram")
        j = normalize_axis(feature, self.n_features_)
        return self._counts[j].copy(), self._edges[j].copy()

    def quantile(self, q) -> np.ndarray:
        """Approximate streaming quantiles per feature.

        Parameters
        ----------
        q : float or array_like of float
            Quantile levels in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_features,)`` for scalar ``q``, else
            ``(len(q), n_features)``. Features with no observations give NaN.

        Raises
        ------
        ValueError
            If no chunk has been seen yet or ``q`` is outside ``[0, 1]``.

        Notes
        -----
        Linear interpolation inside the cumulative histogram, with the
        underflow and overflow counters treated as two extra bins reaching the
        exact running minimum and maximum. The result is therefore always
        within ``[minimum(), maximum()]`` and, for mass inside the grid,
        within one bin width of the exact quantile. Time O(n_features * bins),
        space O(n_features * bins).
        """
        self._require_data("quantile")
        q_arr = as_float_array(q, name="q")
        if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 1).any():
            raise ValueError("q must lie in [0, 1]")
        levels = np.atleast_1d(q_arr).ravel()
        out = np.full((levels.size, self.n_features_), np.nan)
        for j in range(self.n_features_):  # loop over features; inner work is vectorised
            counts = np.concatenate(([self.underflow_[j]], self._counts[j], [self.overflow_[j]]))
            total = counts.sum()
            if total <= 0:
                continue
            lo = np.fmin(self._min[j], self._edges[j, 0])
            hi = np.fmax(self._max[j], self._edges[j, -1])
            edges = np.concatenate(([lo], self._edges[j], [hi]))
            cum = np.cumsum(counts)
            targets = levels * total
            k = np.searchsorted(cum, targets, side="left")
            k = np.clip(k, 0, counts.size - 1)
            prev = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)
            width = counts[k]
            frac = np.where(width > 0, (targets - prev) / np.where(width > 0, width, 1.0), 0.0)
            vals = edges[k] + np.clip(frac, 0.0, 1.0) * (edges[k + 1] - edges[k])
            out[:, j] = np.clip(vals, self._min[j], self._max[j])
        return out[0] if q_arr.ndim == 0 else out

    def _require_data(self, what: str) -> None:
        if self.n_features_ is None or not self._has_edges:
            raise ValueError(
                f"cannot compute {what}: no data has been seen yet; call update_stats first"
            )

    def result(self) -> Dict[str, object]:
        """Every statistic in one dictionary.

        Returns
        -------
        dict
            Keys ``n_samples_seen``, ``count``, ``mean``, ``var``, ``std``,
            ``min``, ``max``, ``q25``, ``median``, ``q75``. The quantile
            entries are approximate (see the class docstring); everything else
            is exact.

        Raises
        ------
        ValueError
            If no chunk has been seen yet.

        Notes
        -----
        Time O(n_features * bins), space O(n_features).
        """
        self._require_data("result")
        q25, med, q75 = self.quantile([0.25, 0.5, 0.75])
        return {
            "n_samples_seen": self.n_samples_seen_,
            "count": self.count,
            "mean": self.mean,
            "var": self.var(),
            "std": self.std(),
            "min": self.minimum(),
            "max": self.maximum(),
            "q25": q25,
            "median": med,
            "q75": q75,
        }

    def __repr__(self) -> str:
        d = "unset" if self.n_features_ is None else self.n_features_
        return (
            f"StreamingStats(bins={self.bins}, n_features_={d}, "
            f"n_samples_seen_={self.n_samples_seen_})"
        )
