"""Decision tree classifier with buffered retraining for streaming data.

The tree follows the classic recursive construction: measure how mixed a node
is, try every candidate question, keep the one with the largest drop in
impurity, split, and repeat until a stopping rule fires. Impurity is either
Gini (``1 - sum p^2``) or entropy (``-sum p log2 p``), and the drop is the
information gain.

Two things distinguish this implementation from the textbook version:

* **Numerical splits.** Questions have the form ``x[:, j] <= threshold``
  rather than ``x[:, j] == 1``. Candidate thresholds are the midpoints
  between consecutive distinct values of a feature, and all of them are
  scored in one vectorised pass over a cumulative class-count matrix.
* **Buffered retraining.** :meth:`DecisionTreeClassifier.partial_fit` lets the
  tree take part in a stream: chunks accumulate in a buffer and the tree is
  rebuilt only when the buffer is large enough, so the cost of rebuilding is
  paid once per ``buffer_size`` samples instead of once per chunk.

Categorical features are supported by one-hot encoding them upstream (see
:class:`numcompute.preprocessing.OneHotEncoder`). A 0/1 indicator column is
split at a threshold of 0.5, which is exactly the ``== 1`` test of a
categorical tree, so no separate categorical code path is needed.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import numpy as np

from numcompute.metrics import accuracy
from numcompute.pipeline import Estimator
from numcompute.utils import check_2d, check_is_fitted

__all__ = [
    "gini_impurity",
    "entropy_impurity",
    "information_gain",
    "DecisionTreeClassifier",
]

#: Gains below this are treated as floating-point noise rather than a real split.
_MIN_GAIN = 1e-12


# --------------------------------------------------------------------------
# Impurity and gain
# --------------------------------------------------------------------------
def _class_proportions(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Row-normalise class counts, returning ``(proportions, totals)``."""
    c = np.asarray(counts, dtype=float)
    if c.ndim == 0:
        raise ValueError("counts must have at least one dimension (a class axis)")
    total = c.sum(axis=-1)
    safe = np.where(total > 0, total, 1.0)
    return c / safe[..., None], total


def gini_impurity(counts) -> Union[float, np.ndarray]:
    """Gini impurity ``1 - sum(p**2)`` of one or many class-count vectors.

    Parameters
    ----------
    counts : array_like, shape (..., n_classes)
        Class counts. The last axis is the class axis; any leading axes are
        treated as independent nodes, which is what lets the split search
        score every candidate threshold at once.

    Returns
    -------
    float or numpy.ndarray
        Impurity with shape ``counts.shape[:-1]``; a scalar for a 1-D input.
        0 for a pure node and for an empty node, ``1 - 1/n_classes`` when the
        classes are balanced.

    Raises
    ------
    ValueError
        If ``counts`` is 0-dimensional.

    Notes
    -----
    Time O(size), space O(size).

    Examples
    --------
    >>> float(gini_impurity([4, 0]))
    0.0
    >>> float(gini_impurity([2, 2]))
    0.5
    """
    p, total = _class_proportions(counts)
    out = np.where(total > 0, 1.0 - (p ** 2).sum(axis=-1), 0.0)
    return float(out) if out.ndim == 0 else out


def entropy_impurity(counts) -> Union[float, np.ndarray]:
    """Shannon entropy ``-sum(p * log2 p)`` of one or many class-count vectors.

    Parameters
    ----------
    counts : array_like, shape (..., n_classes)
        As in :func:`gini_impurity`.

    Returns
    -------
    float or numpy.ndarray
        Impurity in bits, shape ``counts.shape[:-1]``. 0 for a pure or empty
        node, ``log2(n_classes)`` when the classes are balanced.

    Raises
    ------
    ValueError
        If ``counts`` is 0-dimensional.

    Notes
    -----
    The convention ``0 * log2(0) = 0`` is applied, so empty classes
    contribute nothing and no warning is emitted. Time O(size), space
    O(size).

    Examples
    --------
    >>> float(entropy_impurity([4, 0]))
    0.0
    >>> float(entropy_impurity([2, 2]))
    1.0
    """
    p, total = _class_proportions(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    out = np.where(total > 0, -terms.sum(axis=-1), 0.0)
    # Avoid returning a negative signed zero for a pure node.
    out = np.where(out == 0.0, 0.0, out)
    return float(out) if out.ndim == 0 else out


_CRITERIA = {"gini": gini_impurity, "entropy": entropy_impurity}


def information_gain(parent_counts, left_counts, right_counts, criterion: str = "gini"):
    """Drop in impurity produced by splitting a node in two.

    ``gain = I(parent) - (n_left * I(left) + n_right * I(right)) / n``, the
    weighted-average formulation: the bigger child counts for more.

    Parameters
    ----------
    parent_counts : array_like, shape (..., n_classes)
        Class counts before the split. Should equal
        ``left_counts + right_counts``.
    left_counts, right_counts : array_like, shape (..., n_classes)
        Class counts of the two children.
    criterion : {'gini', 'entropy'}, optional
        Impurity measure. Default ``'gini'``.

    Returns
    -------
    float or numpy.ndarray
        Gain with shape given by broadcasting the inputs' leading axes.
        0 when the children are empty. Never negative for a genuine split,
        because both impurity measures are concave.

    Raises
    ------
    ValueError
        If ``criterion`` is unknown or an input is 0-dimensional.

    Notes
    -----
    Time O(size), space O(size).

    Examples
    --------
    >>> float(information_gain([2, 2], [2, 0], [0, 2]))   # a perfect split
    0.5
    >>> float(information_gain([2, 2], [1, 1], [1, 1]))   # a useless one
    0.0
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {tuple(_CRITERIA)}, got {criterion!r}")
    impurity = _CRITERIA[criterion]
    left = np.asarray(left_counts, dtype=float)
    right = np.asarray(right_counts, dtype=float)
    n_left, n_right = left.sum(axis=-1), right.sum(axis=-1)
    total = n_left + n_right
    safe = np.where(total > 0, total, 1.0)
    child = (n_left * impurity(left) + n_right * impurity(right)) / safe
    out = np.where(total > 0, impurity(parent_counts) - child, 0.0)
    return float(out) if np.ndim(out) == 0 else out


# --------------------------------------------------------------------------
# Tree storage
# --------------------------------------------------------------------------
class _Tree:
    """Flat array storage for a fitted tree.

    Arrays are indexed by node id. ``children_left[i] == -1`` marks a leaf,
    in which case ``feature[i]`` is -1 and ``threshold[i]`` is NaN. Keeping
    the tree in arrays rather than linked objects is what allows
    :meth:`DecisionTreeClassifier.predict` to push every sample down one
    level at a time with vectorised indexing.
    """

    __slots__ = ("feature", "threshold", "children_left", "children_right",
                 "value", "impurity", "n_node_samples", "max_depth")

    def __init__(self, feature, threshold, children_left, children_right,
                 value, impurity, n_node_samples, max_depth):
        self.feature = feature
        self.threshold = threshold
        self.children_left = children_left
        self.children_right = children_right
        self.value = value
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.max_depth = max_depth

    @property
    def node_count(self) -> int:
        """Total number of nodes."""
        return int(self.feature.size)

    @property
    def n_leaves(self) -> int:
        """Number of leaves."""
        return int(np.count_nonzero(self.children_left == -1))

    def nbytes(self) -> int:
        """Bytes occupied by the stored arrays."""
        return int(sum(getattr(self, name).nbytes for name in
                       ("feature", "threshold", "children_left", "children_right",
                        "value", "impurity", "n_node_samples")))


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------
class DecisionTreeClassifier(Estimator):
    """Depth-limited decision tree with buffered retraining for streams.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, optional
        Impurity measure driving the split search. Default ``'gini'``.
    max_depth : int, optional
        Maximum depth; the root is depth 0. Default ``None`` (unlimited,
        bounded in practice by the other stopping rules).
    min_samples_split : int, optional
        A node with fewer samples becomes a leaf. Default 2, i.e. ``>= 2``.
    max_features : int, float, {'sqrt', 'log2'} or None, optional
        How many features to consider at each split. ``None`` (default)
        considers all of them and consumes no randomness, so the tree is
        fully deterministic. An int is a count, a float in ``(0, 1]`` a
        fraction, and ``'sqrt'``/``'log2'`` the usual shorthands; all are
        clamped to ``[1, n_features]`` and drawn without replacement using
        ``random_state``.
    buffer_size : int, optional
        Samples that must accumulate in the streaming buffer before the tree
        is rebuilt, ``>= 1``. Default 1, which rebuilds on every non-empty
        chunk. See :meth:`partial_fit`.
    random_state : int or None, optional
        Seed for the per-split feature subsampling. Irrelevant when
        ``max_features`` is ``None``. Default ``None``.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Sorted class labels of the most recent training run.
    tree_ : object
        Fitted tree structure. Present only once training has succeeded.
    n_features_in_ : int
        Features the current tree was trained on.
    n_samples_seen_ : int
        Rows handed to :meth:`fit` and :meth:`partial_fit` in total.
    n_retrains_ : int
        Number of completed tree builds, counting the one performed by
        :meth:`fit`.
    last_fit_seconds_ : float
        Wall-clock seconds the most recent build took.
    last_fit_samples_ : int
        Rows the most recent build used.

    Raises
    ------
    ValueError
        At fit time for an unknown ``criterion``, a non-positive
        ``buffer_size``, ``min_samples_split < 2``, ``max_depth < 0`` or an
        unusable ``max_features``.

    Notes
    -----
    **Complexity.** Building the tree costs O(n d log n) time (a sort per
    feature per node) and O(n k) space for the cumulative class-count matrix,
    for n samples, d features and k classes. Prediction is O(depth) vectorised
    passes over the samples, i.e. O(m * depth) work for m rows.

    **Determinism.** With ``max_features=None`` the result depends only on the
    data: among equally good splits the lowest feature index wins, and within
    a feature the lowest threshold wins; a leaf with tied class counts
    predicts the lowest class label. Nothing depends on dictionary or
    floating-point summation order.

    **NaN.** A NaN never satisfies ``x <= threshold``, so rows with a missing
    value in the split feature go to the right child. Thresholds are proposed
    only from a feature's non-NaN values, and a feature that is entirely NaN
    at a node offers no split there. Infinities are rejected instead, because
    a midpoint involving one is not a usable threshold.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    >>> tree.predict([[0.5], [2.5]]).tolist()
    [0, 1]
    >>> tree.score(X, y)
    1.0
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Union[int, float, str, None] = None,
        buffer_size: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.random_state = random_state
        self._reset_stream()
        self.n_samples_seen_ = 0
        self.n_retrains_ = 0
        self.last_fit_seconds_ = 0.0
        self.last_fit_samples_ = 0

    # ------------------------------------------------------------ validation
    def _check_params(self) -> None:
        if self.criterion not in _CRITERIA:
            raise ValueError(
                f"criterion must be one of {tuple(_CRITERIA)}, got {self.criterion!r}"
            )
        if (isinstance(self.min_samples_split, bool)
                or not isinstance(self.min_samples_split, (int, np.integer))
                or self.min_samples_split < 2):
            raise ValueError(
                f"min_samples_split must be an int >= 2, got {self.min_samples_split!r}"
            )
        if self.max_depth is not None:
            if (isinstance(self.max_depth, bool)
                    or not isinstance(self.max_depth, (int, np.integer))
                    or self.max_depth < 0):
                raise ValueError(f"max_depth must be None or an int >= 0, got {self.max_depth!r}")
        if (isinstance(self.buffer_size, bool)
                or not isinstance(self.buffer_size, (int, np.integer))
                or self.buffer_size < 1):
            raise ValueError(f"buffer_size must be an int >= 1, got {self.buffer_size!r}")

    def _n_features_to_try(self, d: int) -> int:
        """Resolve ``max_features`` against the actual feature count."""
        mf = self.max_features
        if mf is None:
            return d
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(d)))
            if mf == "log2":
                return max(1, int(np.log2(d))) if d > 1 else 1
            raise ValueError(f"max_features must be 'sqrt', 'log2', an int, a float or None, got {mf!r}")
        if isinstance(mf, bool):
            raise ValueError("max_features must not be a bool")
        if isinstance(mf, (int, np.integer)):
            if mf < 1:
                raise ValueError(f"max_features as an int must be >= 1, got {mf}")
            return min(int(mf), d)
        if isinstance(mf, (float, np.floating)):
            if not 0.0 < mf <= 1.0:
                raise ValueError(f"max_features as a float must lie in (0, 1], got {mf}")
            return max(1, min(d, int(round(float(mf) * d))))
        raise ValueError(f"max_features must be 'sqrt', 'log2', an int, a float or None, got {mf!r}")

    @staticmethod
    def _check_X(X, name: str = "X") -> np.ndarray:
        X = check_2d(X, name)
        if np.isinf(X).any():
            raise ValueError(
                f"{name} contains an infinity; a split threshold is the midpoint of two "
                f"values, which is not meaningful for infinities. Clip them first."
            )
        return X

    @staticmethod
    def _check_y(y, n_rows: int) -> np.ndarray:
        arr = np.asarray(y)
        if arr.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {arr.shape}")
        if arr.shape[0] != n_rows:
            raise ValueError(f"y must have {n_rows} entries to match X, got {arr.shape[0]}")
        if arr.dtype.kind == "O":
            raise TypeError("y must hold numbers, strings or booleans, not objects")
        if arr.dtype.kind == "f" and np.isnan(arr).any():
            raise ValueError("y contains NaN; labels must be known for every training row")
        return arr

    def _check_xy(self, X, y, allow_empty: bool) -> Tuple[np.ndarray, np.ndarray]:
        X = self._check_X(X)
        y = self._check_y(y, X.shape[0])
        if X.shape[0] == 0 and not allow_empty:
            raise ValueError("cannot train on an empty dataset: X has 0 rows")
        if X.shape[0] and X.shape[1] == 0:
            raise ValueError("cannot train on a dataset with 0 features")
        return X, y

    # ------------------------------------------------------------- training
    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Build a tree from scratch on ``(X, y)``.

        Any streaming state is discarded first, so ``fit`` is a clean restart
        rather than an addition to what :meth:`partial_fit` has buffered.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data. NaN is allowed and routed right; infinities are not.
        y : array_like, shape (n_samples,)
            Class labels; any orderable dtype except object.

        Returns
        -------
        DecisionTreeClassifier
            ``self``.

        Raises
        ------
        TypeError
            If ``X`` is not numeric or ``y`` has object dtype.
        ValueError
            If a parameter is invalid, ``X`` is empty or not 2-D, ``y`` does
            not line up with ``X``, or either contains a forbidden value.

        Notes
        -----
        Time O(n d log n), space O(n k).
        """
        self._check_params()
        X, y = self._check_xy(X, y, allow_empty=False)
        self._train(X, y)  # commit first, so a failed build loses nothing
        self._reset_stream()
        self.n_samples_seen_ = int(X.shape[0])
        return self

    def partial_fit(self, X_chunk, y_chunk) -> "DecisionTreeClassifier":
        """Add one chunk to the buffer, rebuilding the tree when it is full.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
            One chunk of the stream. Zero rows is allowed and does nothing.
        y_chunk : array_like, shape (n_samples,)
            Labels for the chunk.

        Returns
        -------
        DecisionTreeClassifier
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError, ValueError
            As in :meth:`fit`, plus a ``ValueError`` if the chunk's feature
            count differs from what is already buffered or trained. The check
            happens before anything is appended, so a rejected chunk leaves
            the buffer untouched.

        Notes
        -----
        **The policy, exactly.** Every call validates the chunk and appends it
        to an internal buffer. While the buffer holds fewer than
        ``buffer_size`` samples nothing else happens, and :meth:`predict`
        keeps using the tree from the previous rebuild. As soon as the buffer
        holds ``buffer_size`` or more, the tree is rebuilt **on the buffered
        samples only** and the buffer is emptied. The tree therefore tracks
        recent data, which is the point of a streaming learner, rather than
        growing without bound.

        **Oversized chunks.** A single chunk larger than ``buffer_size``
        triggers exactly one rebuild, on the whole buffer, and empties it. The
        buffer is never left holding a remainder and a chunk is never split,
        so at most one rebuild happens per call however big the chunk is.

        **Failure.** The new tree is built into local variables and only
        assigned to the instance once it is complete. If a rebuild raises, the
        previously trained tree stays usable *and* the buffer keeps its
        samples, so nothing is lost and the next call can try again.

        Time O(1) for a call that only buffers; the cost of :meth:`fit` on
        ``buffer_size`` samples for a call that rebuilds. Space O(buffered
        rows * n_features).
        """
        self._check_params()
        X, y = self._check_xy(X_chunk, y_chunk, allow_empty=True)
        if X.shape[0]:
            expected = self._buffered_width()
            if expected is not None and X.shape[1] != expected:
                raise ValueError(
                    f"X_chunk has {X.shape[1]} features, but this tree is working with "
                    f"{expected}"
                )
            self._buffer_X.append(X)
            self._buffer_y.append(y)
            self._buffer_n += int(X.shape[0])
            self.n_samples_seen_ += int(X.shape[0])
        if self._buffer_n >= self.buffer_size:
            X_buf = np.vstack(self._buffer_X)
            y_buf = np.concatenate(self._buffer_y)
            self._train(X_buf, y_buf)  # raises before committing, leaving the buffer intact
            self._reset_stream()
        return self

    def _reset_stream(self) -> None:
        self._buffer_X: List[np.ndarray] = []
        self._buffer_y: List[np.ndarray] = []
        self._buffer_n = 0

    def _buffered_width(self) -> Optional[int]:
        """Feature count the tree is currently committed to, or None."""
        if self._buffer_X:
            return int(self._buffer_X[0].shape[1])
        return getattr(self, "n_features_in_", None)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build a tree and commit it atomically."""
        started = time.perf_counter()
        classes, y_idx = np.unique(y, return_inverse=True)
        tree = self._grow(X, np.asarray(y_idx, dtype=np.intp).ravel(), classes.size)
        elapsed = time.perf_counter() - started
        # Nothing above touched self, so a failure leaves the old tree in place.
        self.classes_ = classes
        self.tree_ = tree
        self.n_features_in_ = int(X.shape[1])
        self.n_retrains_ += 1
        self.last_fit_seconds_ = elapsed
        self.last_fit_samples_ = int(X.shape[0])

    def _grow(self, X: np.ndarray, y_idx: np.ndarray, n_classes: int) -> _Tree:
        """Grow the tree with an explicit stack.

        The algorithm is the usual recursive one; the stack replaces Python
        recursion so that a pathologically deep tree cannot raise
        ``RecursionError``.
        """
        impurity_fn = _CRITERIA[self.criterion]
        rng = np.random.default_rng(self.random_state)
        depth_cap = np.inf if self.max_depth is None else int(self.max_depth)
        n_try = self._n_features_to_try(X.shape[1])

        feature: List[int] = []
        threshold: List[float] = []
        left: List[int] = []
        right: List[int] = []
        value: List[np.ndarray] = []
        impurity: List[float] = []
        n_node: List[int] = []
        deepest = 0

        # (row indices, depth, parent id, came from the parent's left branch)
        stack: List[Tuple[np.ndarray, int, int, bool]] = [
            (np.arange(X.shape[0], dtype=np.intp), 0, -1, False)
        ]
        while stack:
            rows, depth, parent, is_left = stack.pop()
            counts = np.bincount(y_idx[rows], minlength=n_classes).astype(float)
            node_id = len(feature)
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(counts)
            impurity.append(float(impurity_fn(counts)))
            n_node.append(int(rows.size))
            deepest = max(deepest, depth)
            if parent >= 0:
                (left if is_left else right)[parent] = node_id

            if (depth >= depth_cap
                    or rows.size < self.min_samples_split
                    or impurity[node_id] <= 0.0):
                continue  # stopping rule fired: this node stays a leaf
            split = self._best_split(X, y_idx, rows, counts, n_classes, n_try, rng)
            if split is None:
                continue
            j, thr = split
            go_left = X[rows, j] <= thr  # NaN compares False, so it goes right
            if not go_left.any() or go_left.all():
                continue  # float rounding collapsed the split; keep a leaf
            feature[node_id] = j
            threshold[node_id] = thr
            # push right first so the left child is processed next (pre-order)
            stack.append((rows[~go_left], depth + 1, node_id, False))
            stack.append((rows[go_left], depth + 1, node_id, True))

        return _Tree(
            feature=np.asarray(feature, dtype=np.intp),
            threshold=np.asarray(threshold, dtype=float),
            children_left=np.asarray(left, dtype=np.intp),
            children_right=np.asarray(right, dtype=np.intp),
            value=np.asarray(value, dtype=float).reshape(len(value), n_classes),
            impurity=np.asarray(impurity, dtype=float),
            n_node_samples=np.asarray(n_node, dtype=np.intp),
            max_depth=deepest,
        )

    def _best_split(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        rows: np.ndarray,
        counts: np.ndarray,
        n_classes: int,
        n_try: int,
        rng: np.random.Generator,
    ) -> Optional[Tuple[int, float]]:
        """Best ``(feature, threshold)`` at one node, or None if nothing helps.

        Every candidate threshold of a feature is scored in one pass: sort the
        feature, build the cumulative class counts of the sorted prefix, and
        read off the left child's counts for all splits at once. The right
        child is the parent minus the left, which automatically sends rows
        with a NaN in this feature to the right.

        Candidates are the midpoints between consecutive distinct observed
        values, plus — when the node holds rows whose value is missing — the
        largest observed value itself. That extra candidate is what lets the
        tree ask "is this value missing?", which the midpoints alone cannot
        express, since no midpoint ever lies above the largest observation.
        """
        d = X.shape[1]
        if n_try >= d:
            candidates = np.arange(d)
        else:
            # sorted so that ties are broken by the lowest feature index
            candidates = np.sort(rng.choice(d, size=n_try, replace=False))

        parent_impurity = float(_CRITERIA[self.criterion](counts))
        total = counts
        n_rows = float(rows.size)
        best_gain, best = _MIN_GAIN, None

        for j in candidates:  # loop over features; all thresholds are vectorised
            col = X[rows, j]
            seen = ~np.isnan(col)
            n_seen = int(seen.sum())
            n_missing = int(rows.size) - n_seen
            if n_seen == 0:
                continue  # every value of this feature is missing here
            values = col[seen]
            order = np.argsort(values, kind="stable")
            ordered = values[order]
            boundary = ordered[1:] != ordered[:-1]
            if not boundary.any() and n_missing == 0:
                continue  # constant feature with nothing missing: no split separates anything
            labels = y_idx[rows[seen]][order]
            onehot = np.zeros((n_seen, n_classes))
            onehot[np.arange(n_seen), labels] = 1.0
            cumulative = np.cumsum(onehot, axis=0)
            prefix = cumulative[:-1][boundary]  # left child counts, one row per midpoint
            lows = ordered[:-1][boundary]
            highs = ordered[1:][boundary]
            if n_missing:
                # "observed goes left, missing goes right": a threshold at the maximum
                prefix = np.vstack([prefix, cumulative[-1][None, :]])
                lows = np.append(lows, ordered[-1])
                highs = np.append(highs, ordered[-1])
            child_right = total - prefix
            n_left = prefix.sum(axis=1)
            n_right = n_rows - n_left
            gains = parent_impurity - (
                n_left * _CRITERIA[self.criterion](prefix)
                + n_right * _CRITERIA[self.criterion](child_right)
            ) / n_rows
            best_here = int(np.argmax(gains))  # argmax keeps the first, i.e. lowest threshold
            if gains[best_here] > best_gain:
                lo, hi = lows[best_here], highs[best_here]
                best_gain = float(gains[best_here])
                best = (int(j), float(lo + (hi - lo) / 2.0))
        return best

    # ------------------------------------------------------------ prediction
    def _apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf id reached by each row, found one level at a time."""
        t = self.tree_
        node = np.zeros(X.shape[0], dtype=np.intp)
        while True:
            internal = t.children_left[node] != -1
            if not internal.any():
                return node
            moving = np.nonzero(internal)[0]
            here = node[moving]
            go_left = X[moving, t.feature[here]] <= t.threshold[here]
            node[moving] = np.where(go_left, t.children_left[here], t.children_right[here])

    def _check_predict_input(self, X) -> np.ndarray:
        check_is_fitted(self, "tree_")
        X = self._check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was trained with "
                f"{self.n_features_in_}"
            )
        return X

    def predict(self, X) -> np.ndarray:
        """Predict a class label for every row of ``X``.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data to classify. NaN is allowed and routed right.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Labels drawn from :attr:`classes_`.

        Raises
        ------
        NotFittedError
            If no tree has been built yet — that is, before :meth:`fit`, or
            before :meth:`partial_fit` has first filled the buffer. Predicting
            from a model that has never been trained has no defined answer, so
            it is reported rather than guessed at.
        ValueError
            If ``X`` is not 2-D, has the wrong number of features, or contains
            an infinity.

        Notes
        -----
        Between rebuilds this uses the most recently committed tree. Ties in a
        leaf's class counts resolve to the lowest label in :attr:`classes_`.
        Time O(n_samples * depth), space O(n_samples).
        """
        X = self._check_predict_input(X)
        leaves = self._apply(X)
        return self.classes_[np.argmax(self.tree_.value[leaves], axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, taken from the class mix of the reached leaf.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
            Rows sum to 1, in the order of :attr:`classes_`.

        Raises
        ------
        NotFittedError, ValueError
            As in :meth:`predict`.

        Notes
        -----
        These are training frequencies, not calibrated probabilities: a pure
        leaf reports 1.0 however few samples it holds. They are good enough to
        rank samples, which is what :class:`numcompute.metrics.StreamingAUC`
        needs. Time O(n_samples * depth), space O(n_samples * n_classes).
        """
        X = self._check_predict_input(X)
        counts = self.tree_.value[self._apply(X)]
        totals = counts.sum(axis=1, keepdims=True)
        return counts / np.where(totals > 0, totals, 1.0)

    def score(self, X, y) -> float:
        """Mean accuracy on ``(X, y)``.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction correct, in ``[0, 1]``.

        Raises
        ------
        NotFittedError, ValueError
            As in :meth:`predict`.

        Notes
        -----
        Delegates to :func:`numcompute.metrics.accuracy`.
        """
        return float(accuracy(y, self.predict(X)))

    # ------------------------------------------------------------ inspection
    @property
    def is_fitted_(self) -> bool:
        """True once a tree has been built and committed."""
        return hasattr(self, "tree_")

    @property
    def n_buffered_(self) -> int:
        """Samples waiting in the streaming buffer."""
        return self._buffer_n

    @property
    def depth_(self) -> int:
        """Depth of the fitted tree (0 for a single leaf)."""
        check_is_fitted(self, "tree_")
        return int(self.tree_.max_depth)

    @property
    def n_nodes_(self) -> int:
        """Number of nodes in the fitted tree."""
        check_is_fitted(self, "tree_")
        return self.tree_.node_count

    @property
    def n_leaves_(self) -> int:
        """Number of leaves in the fitted tree."""
        check_is_fitted(self, "tree_")
        return self.tree_.n_leaves

    def memory_bytes(self) -> int:
        """Bytes held by the fitted tree plus the streaming buffer.

        Returns
        -------
        int
            Sum of ``nbytes`` over the tree's arrays and the buffered chunks.
            This measures the data the model actually owns; it excludes Python
            object overhead, which ``sys.getsizeof`` would only partly capture
            anyway.

        Notes
        -----
        Time O(number of buffered chunks), space O(1).
        """
        buffered = sum(a.nbytes for a in self._buffer_X) + sum(a.nbytes for a in self._buffer_y)
        return int(buffered + (self.tree_.nbytes() if self.is_fitted_ else 0))

    def __repr__(self) -> str:
        state = (
            f"fitted, depth={self.depth_}, leaves={self.n_leaves_}"
            if self.is_fitted_ else "not fitted"
        )
        return (
            f"DecisionTreeClassifier(criterion={self.criterion!r}, max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, max_features={self.max_features!r}, "
            f"buffer_size={self.buffer_size}) [{state}, buffered={self.n_buffered_}, "
            f"retrains={self.n_retrains_}]"
        )
