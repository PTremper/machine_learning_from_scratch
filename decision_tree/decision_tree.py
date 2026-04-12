"""Implements a Decision Tree from scratch in pure python plus numpy.

This implementation is based on:
- https://www.youtube.com/watch?v=NxEHSAfFlK8&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=7
- https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/blob/main/04%20Decision%20Trees/DecisionTree.py

"""

from __future__ import annotations

from collections import Counter
from typing import Self

import numpy as np
from numpy.typing import NDArray


class Node:
    """
    A single node in the decision tree.

    A node can either be:
    - an internal node (contains a split rule), or
    - a leaf node (contains a predicted class value)
    """

    def __init__(
        self,
        feature_index: int | None = None,
        threshold: float | None = None,
        left: Self | None = None,
        right: Self | None = None,
        *,
        value: int | None = None,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # If value is set, this is a leaf node
        self.value = value

    def is_leaf(self) -> bool:
        """Return True if this node is a leaf (i.e. has a prediction)."""
        return self.value is not None


class DecisionTree:
    """A simple decision tree classifier.

    The tree works by repeatedly asking yes/no questions of the form:

        "Is feature_j <= threshold?"

    Each question splits the dataset into two parts. The goal is to make
    those parts as "pure" as possible (i.e. containing mostly one class).

    Over time, the data is split into smaller and smaller subsets until
    we stop and assign a class label to each final subset (leaf node).

    Methods
    -------
    fit(X, y)
        Fit the decision tree to the training data.
    predict(X)
        Predict the class labels for the input data.

    """

    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        max_features: int | None = None,
    ) -> None:
        """Initialize the decision tree classifier.

        Parameters
        ----------
        min_samples_split: int, default=2
            Minimum number of samples required to keep splitting.
            If a node has fewer samples than this, it becomes a leaf.

        max_depth: int, default=100
            Maximum depth of the tree. Prevents infinite growth.

        max_features: int | None, default=None
            Number of features (dimensions) to consider at each split.
            If None, all features are used (not a random subset).
        """
        self.rng = np.random.default_rng()

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features

        self.root: Node | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.integer]) -> None:
        """Build the decision tree from the training data.

        X: shape (n_samples, n_features)
        y: shape (n_samples,)

        Each row in X is a sample. Each column is a feature (dimension).
        """
        n_features_total = X.shape[1]

        if self.max_features is None:
            self.max_features = n_features_total
        else:
            self.max_features = min(n_features_total, self.max_features)

        self.root = self._grow_tree(X, y)

    def _grow_tree(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.integer],
        depth: int = 0,
    ) -> Node:
        """Recursively build the tree.

        At each step:
        - check if we should stop
        - otherwise find the best split
        - split the data
        - recurse on both sides
        """
        n_samples, n_features_total = X.shape
        n_classes = len(np.unique(y))

        # stopping conditions: if true, return node as leaf node
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # choose random subset of features (dimensions)
        feature_indices = self.rng.choice(
            n_features_total,
            self.max_features,
            replace=False,
        )

        # find best split among those features
        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        # split dataset
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)

        # recursively grow children
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def _best_split(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.integer],
        feature_indices: NDArray[np.integer],
    ) -> tuple[int, float]:
        """
        Find the best (feature, threshold) pair to split on.

        Strategy:
        - loop over selected features
        - for each feature, try all possible thresholds
        - pick the split with highest information gain
        """
        best_gain = -1.0  # -1 is overwritten by any valid gain as entropy is always >= 0
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, feature_values, threshold)

                if gain > best_gain:
                    best_gain = gain
                    # explictly type casting to pacify the type checker
                    # type checker can't infer the type of feat_idx and thr in the loop
                    # casting numpy types to built-in types is fine here
                    best_feature = int(feature_idx)
                    best_threshold = float(threshold)

        if best_feature is None or best_threshold is None:
            msg = "Failed to find a valid split."
            raise AssertionError(msg)

        return best_feature, best_threshold

    def _information_gain(
        self,
        y: NDArray[np.integer],
        feature_values: NDArray[np.float64],
        threshold: float,
    ) -> float:
        """Measure how much a split improves purity.

        High information gain = good split.
        """
        parent_entropy = self._entropy(y)

        left_indices, right_indices = self._split(feature_values, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0.0

        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        # calculate the weighted avg. entropy of children
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        # information gain is the reduction in entropy
        return parent_entropy - child_entropy

    def _split(
        self,
        feature_values: NDArray[np.float64],
        threshold: float,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        """
        Split indices based on a threshold.

        Left:  value <= threshold
        Right: value > threshold
        """
        left_indices = np.argwhere(feature_values <= threshold).flatten()
        right_indices = np.argwhere(feature_values > threshold).flatten()
        return left_indices, right_indices

    def _entropy(self, y: NDArray[np.integer]) -> float:
        """Measure how mixed the labels are.

        - 0 → perfectly pure (only one class)
        - higher → more mixed
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    def _most_common_label(self, y: NDArray[np.integer]) -> int:
        """Return the most frequent class in y."""
        return Counter(y).most_common(1)[0][0]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.integer]:
        """Predict class labels for each sample in X.

        Each sample is passed through the tree until a leaf is reached.
        """
        if self.root is None:
            msg = "The decision tree has not been fitted yet. Call fit() before predict()."
            raise AssertionError(msg)

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: NDArray[np.float64], node: Node) -> int:
        """Follow the decision path for a single sample."""
        # check if this is a leaf node (i.e., has a value)
        if node.value is not None:
            return node.value

        if (
            node.feature_index is None
            or node.threshold is None
            or node.left is None
            or node.right is None
        ):
            msg = (
                "Invalid tree structure. "
                "Non-leaf nodes should have feature, threshold, left, right set."
                "Check method _grow_tree."
            )
            raise AssertionError(msg)

        # if it is not a leaf node, traverse to the next node
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
