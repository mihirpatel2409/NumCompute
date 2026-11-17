"""Streaming statistics.

Re-exports the statistics layer of :mod:`numcompute.stats`:
:class:`StreamingStats` (chunked mean, variance, standard deviation,
quantiles and histograms behind one ``update_stats`` call),
:class:`RunningStats` (the exact Welford/Chan engine it is built on) and the
batch functions the two are checked against.
"""
from __future__ import annotations

from numcompute.stats import (
    RunningStats,
    StreamingStats,
    describe,
    histogram,
    maximum,
    mean,
    median,
    minimum,
    quantile,
    std,
    var,
)

__all__ = [
    "StreamingStats",
    "RunningStats",
    "mean",
    "median",
    "var",
    "std",
    "minimum",
    "maximum",
    "describe",
    "histogram",
    "quantile",
]
