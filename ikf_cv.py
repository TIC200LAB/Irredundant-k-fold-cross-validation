"""Irredundant k-fold cross-validation with a scikit-learn-compatible API.

This module implements the IkF splitters described in the manuscript
"Irredundant k-fold cross-validation for reliable, scalable, and
energy-efficient model evaluation".

The main class, :class:`IStratifiedKFold`, intentionally mirrors
``sklearn.model_selection.StratifiedKFold``:

    >>> from ikf_cv import IStratifiedKFold
    >>> iskf = IStratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    >>> for train_idx, test_idx in iskf.split(X, y, rand=True):
    ...     ...

Compatibility policy
--------------------
* The constructor is exactly the public StratifiedKFold constructor.
* Outer test folds are produced by the parent StratifiedKFold implementation.
* Inner (k-1) subfolds are also produced by StratifiedKFold rather than by a
  private/reimplemented sklearn algorithm.
* ``shuffle`` and ``random_state`` therefore retain sklearn semantics.
* ``rand`` is the single IkF extension to ``split``.  It controls only the
  assignment of already-created subfolds to irredundant training sets.

"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_random_state, column_or_1d

__all__ = [
    "IStratifiedKFold",
    "IKFold",
    "IrredundantStratifiedKFold",
    "IrredundantKFold",
]


def _maximum_fully_stratified_k(min_class_size: int) -> int:
    """Largest integer k satisfying k(k-1) <= min_class_size."""
    return int(np.floor((1.0 + np.sqrt(1.0 + 4.0 * min_class_size)) / 2.0))


def _as_int_index(a) -> np.ndarray:
    """Return a one-dimensional sklearn-style integer index array."""
    return np.asarray(a, dtype=np.intp).reshape(-1)


def _legacy_remaining(k_minus_1: int, used: Sequence[int]) -> List[int]:
    """Available subfold labels in the order used by the reference code.
    """
    return list(set(range(k_minus_1)) - set(used))


def _assemble_reference_assignment(
    subfolds: Sequence[Sequence[np.ndarray]],
    *,
    rand: bool,
    rng,
) -> List[List[np.ndarray]]:
    """Assemble IkF training sets using the reference mapping.
    """
    k = len(subfolds)
    n_sub = k - 1
    used = {source: [] for source in range(k)}
    train_parts: List[List[np.ndarray]] = [[] for _ in range(k)]

    # Historical loop: i=0..k-1, skip source=(k-1)-i, then flip train_i.
    for historical_i in range(k):
        dest = (k - 1) - historical_i
        for source in range(k):
            if source == dest:
                continue

            if rand:
                choices = _legacy_remaining(n_sub, used[source])
                sub_idx = int(rng.choice(choices))
            else:
                choices = _legacy_remaining(n_sub, used[source])
                sub_idx = int(choices[0])

            used[source].append(sub_idx)
            train_parts[dest].append(_as_int_index(subfolds[source][sub_idx]))

    return train_parts


def _assemble_reference_deterministic(
    subfolds: Sequence[Sequence[np.ndarray]],
) -> List[List[np.ndarray]]:
    """Exact deterministic (rand=False) mapping of the reference code."""
    k = len(subfolds)
    n_sub = k - 1
    built: List[List[np.ndarray]] = []


    flat = [part for source in subfolds for part in source]
    for i in range(k):
        parts: List[np.ndarray] = []
        for j in range(n_sub):
            if i == n_sub:
                idx = i + (n_sub - 1) + j * n_sub
            else:
                if j >= n_sub - i:
                    idx = i + j * n_sub + (n_sub - 1)
                else:
                    idx = i + j * n_sub
            parts.append(_as_int_index(flat[idx]))
        built.append(parts)

    return list(reversed(built))


class IStratifiedKFold(StratifiedKFold):
    """Irredundant Stratified K-Fold cross-validator.

    Parameters
    ----------
    n_splits : int, default=5
        Number of outer folds. Must be at least 2.
    shuffle : bool, default=False
        Same meaning as in :class:`sklearn.model_selection.StratifiedKFold`.
    random_state : int, RandomState instance or None, default=None
        Same meaning as in StratifiedKFold.  When ``rand=True``, it also
        controls the random assignment of inner subfolds to IkF training sets.

    Notes
    -----
    Each outer fold is used once for testing.  Each outer fold is partitioned
    into ``n_splits - 1`` stratified subfolds.  For a given test fold, the
    training set contains exactly one unused subfold from every other outer
    fold.  Hence each observation is used exactly once for testing and exactly
    once for training across the complete IkF run, and the IkF training sets
    are pairwise disjoint.

    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        # Let sklearn validate constructor semantics (including the rule that
        # random_state must be None when shuffle=False).
        super().__init__(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(self, X, y, groups=None, rand=False):
        """Generate irredundant train/test indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.  Values are not inspected by IkF itself.
        y : array-like of shape (n_samples,)
            Class labels.
        groups : object, default=None
            Passed to the parent splitter for API compatibility; ignored by
            StratifiedKFold, as in scikit-learn.
        rand : bool, default=False
            If False, use the deterministic subfold-allocation policy of the
            reference implementation.  If True, choose one still-available
            subfold from every non-test outer fold at random, without
            replacement.

        Yields
        ------
        train_index : ndarray of shape (n_train,), dtype np.intp
        test_index : ndarray of shape (n_test,), dtype np.intp
        """
        # Parent splitter performs sklearn's standard input validation and
        # determines the test folds.  
        outer_folds = [
            _as_int_index(test_idx)
            for _, test_idx in super().split(X, y, groups)
        ]

        y_arr = column_or_1d(y)
        n = int(len(y_arr))
        k = int(self.n_splits)
        n_sub = k - 1

        if n < k * n_sub:
            raise ValueError(
                f"IkF with n_splits={k} requires at least k(k-1)={k*n_sub} "
                f"samples; got n_samples={n}."
            )

        _, counts = np.unique(y_arr, return_counts=True)
        min_class = int(np.min(counts))
        maximum_k = _maximum_fully_stratified_k(min_class)
        if k > maximum_k:
            warnings.warn(
                "The selected n_splits does not guarantee representation of "
                "every class in every inner subfold. The largest k satisfying "
                f"min_class >= k(k-1) is {maximum_k}.",
                UserWarning,
            )

        subfolds: List[List[np.ndarray]] = []
        last_inner_rng = None

        for outer in outer_folds:
            local_y = y_arr[outer]
            dummy_X = np.zeros((len(outer), 1), dtype=np.uint8)

            if self.shuffle:
                inner_rng = check_random_state(self.random_state)
                inner = StratifiedKFold(
                    n_splits=n_sub,
                    shuffle=True,
                    random_state=inner_rng,
                )
                local_parts = [
                    outer[_as_int_index(local_test)]
                    for _, local_test in inner.split(dummy_X, local_y)
                ]
                last_inner_rng = inner_rng
            else:
 
                inner = StratifiedKFold(
                    n_splits=n_sub,
                    shuffle=False,
                    random_state=None,
                )
                local_parts = [
                    outer[_as_int_index(local_test)]
                    for _, local_test in inner.split(dummy_X, local_y)
                ]

            subfolds.append([_as_int_index(p) for p in local_parts])

        if bool(rand):
            assignment_rng = (
                last_inner_rng
                if self.shuffle
                else check_random_state(None)
            )
            train_parts = _assemble_reference_assignment(
                subfolds,
                rand=True,
                rng=assignment_rng,
            )
        else:
            train_parts = _assemble_reference_deterministic(subfolds)

        for i in range(k):
            train = np.sort(np.concatenate(train_parts[i])).astype(
                np.intp, copy=False
            )
            test = np.sort(outer_folds[i]).astype(np.intp, copy=False)
            yield train, test

    def check_fold(self, folds, y) -> bool:
        """Return True iff every supplied fold contains every class."""
        y_arr = column_or_1d(y)
        classes = np.unique(y_arr)
        for fold in folds:
            if not np.array_equal(np.unique(y_arr[_as_int_index(fold)]), classes):
                return False
        return True


class IKFold(KFold):
    """Non-stratified irredundant K-Fold cross-validator.

    This is the non-stratified analogue of :class:`IStratifiedKFold`.  Its
    public constructor mirrors :class:`sklearn.model_selection.KFold` and its
    ``split`` method adds only the optional ``rand`` argument.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(self, X, y=None, groups=None, rand=False):
        outer_folds = [
            _as_int_index(test_idx)
            for _, test_idx in super().split(X, y, groups)
        ]
        n = int(sum(len(f) for f in outer_folds))
        k = int(self.n_splits)
        n_sub = k - 1

        if n < k * n_sub:
            raise ValueError(
                f"IkF with n_splits={k} requires at least k(k-1)={k*n_sub} "
                f"samples; got n_samples={n}."
            )

        subfolds: List[List[np.ndarray]] = []
        last_inner_rng = None
        for outer in outer_folds:
            dummy_X = np.zeros((len(outer), 1), dtype=np.uint8)
            if self.shuffle:
                inner_rng = check_random_state(self.random_state)
                inner = KFold(
                    n_splits=n_sub,
                    shuffle=True,
                    random_state=inner_rng,
                )
                local_parts = [
                    outer[_as_int_index(local_test)]
                    for _, local_test in inner.split(dummy_X)
                ]
                last_inner_rng = inner_rng
            else:
                inner = KFold(n_splits=n_sub, shuffle=False, random_state=None)
                local_parts = [
                    outer[_as_int_index(local_test)]
                    for _, local_test in inner.split(dummy_X)
                ]
            subfolds.append([_as_int_index(p) for p in local_parts])

        if bool(rand):
            assignment_rng = (
                last_inner_rng if self.shuffle else check_random_state(None)
            )
            train_parts = _assemble_reference_assignment(
                subfolds, rand=True, rng=assignment_rng
            )
        else:
            train_parts = _assemble_reference_deterministic(subfolds)

        for i in range(k):
            train = np.sort(np.concatenate(train_parts[i])).astype(
                np.intp, copy=False
            )
            test = np.sort(outer_folds[i]).astype(np.intp, copy=False)
            yield train, test


# Descriptive aliases.
IrredundantStratifiedKFold = IStratifiedKFold
IrredundantKFold = IKFold
