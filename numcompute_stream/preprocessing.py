"""Incremental preprocessing transformers.

These are the Assignment 2.1 transformers extended in place with
``partial_fit``, re-exported here so the streaming package presents them
under its own namespace. There is one implementation, in
:mod:`numcompute.preprocessing`, so batch and streaming code cannot drift
apart; see that module for the running-statistics and category-growth rules.
"""
from __future__ import annotations

from numcompute.preprocessing import (
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

__all__ = ["StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder"]
