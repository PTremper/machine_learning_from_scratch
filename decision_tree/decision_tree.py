"""Implements a Decision Tree from scratch in pure python plus numpy.

This implementation is based on:
- https://www.youtube.com/watch?v=NxEHSAfFlK8&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=7
- https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/blob/main/04%20Decision%20Trees/DecisionTree.py

"""

from collections import Counter
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray


class Node:
    """A node in the decision tree."""

    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: Self | None = None,
        right: Self | None = None,
        *,
        value: Literal[0, 1] | None = None,
    ) -> None:
        """Initialize a node in the decision tree.

        Parameters
        ----------
        feature : int, optional
            The index of the feature to split on, by default None.
        threshold : float, optional
            The threshold value for the split, by default None.
        left : Node | None, optional
            The left child node, by default None.
        right : Node | None, optional
            The right child node, by default None.
        value : Literal[0, 1] | None, optional
            The value of the node if it is a leaf node, by default None.

        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """Return True if the node is a leaf node (if its value is not None)."""
        return self.value is not None


class DecisionTree:
    """A decision tree classifier.

    Parameters
    ----------
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node, by default 2.
    max_depth : int, optional
        The maximum depth of the tree, by default 100.
    n_features : int | None, optional
        The number of features to consider when looking for the best split, by default None.

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
        n_features: int | None = None,
    ) -> None:
        """Initialize the decision tree classifier.

        Parameters
        ----------
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node, by default 2.
        max_depth : int, optional
            The maximum depth of the tree, by default 100.
        n_features : int | None, optional
            The number of features to consider when looking for the best split, by default None.

        """
        self.rng = np.random.default_rng()

        self.min_samples_split: int = min_samples_split
        self.max_depth: int = max_depth
        self.n_features: int | None = n_features
        self.root: Node | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.integer]) -> None:
        """Fit the decision tree to the training data.

        Parameters
        ----------
        X : NDArray[np.float64]
            The input features.
        y : NDArray[np.integer]
            The target labels.

        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.integer],
        depth: int = 0,
    ) -> Node:
        """Grow a decision tree recursively.

        Parameters
        ----------
        X : NDArray[np.float64]
            The input features.
        y : NDArray[np.integer]
            The target labels.
        depth : int, optional
            The current depth of the tree, by default 0. Restricted by self.max_depth.

        Returns
        -------
        Node
            The root node of the tree.

        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            # if the stopping criteria are met, return a leaf node with the most common label as value
            return Node(value=leaf_value)

        feat_idxs: NDArray[np.integer] = self.rng.choice(
            n_feats,
            self.n_features,
            replace=False,
        )

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.integer],
        feat_idxs: NDArray[np.integer],
    ) -> tuple[int, float]:
        """Calculate the best feature split based on highest information gain.

        Parameters
        ----------
        X : NDArray[np.float64]
            The input features.
        y : NDArray[np.integer]
            The target labels.
        feat_idxs : NDArray[np.integer]
            The indices of the features to consider for splitting.

        Returns
        -------
        tuple[int, float]
            The index of the best feature and the best threshold for the split.

        """
        best_gain: float = -1  # -1 is overwritten by any valid gain as entropy is always >= 0
        split_idx: int | None = None
        split_threshold: float | None = None

        if not feat_idxs.size:
            msg = "No features to consider for splitting."
            raise ValueError(msg)

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    # explictly type casting to pacify the type checker
                    # type checker can't infer the type of feat_idx and thr in the loop
                    # casting numpy types to built-in types is fine here
                    split_idx = int(feat_idx)
                    split_threshold = float(thr)

        if split_idx is None or split_threshold is None:
            msg = "split_idx and split_threshold should be set by the loop above."
            raise AssertionError(msg)

        return split_idx, split_threshold

    def _information_gain(
        self,
        y: NDArray[np.integer],
        X_column: NDArray[np.float64],
        threshold: float,
    ) -> float:
        """Calculate the information gain for a given split threshold.

        Parameters
        ----------
        y : NDArray[np.integer]
            The target variable.
        X_column : NDArray[np.float64]
            The feature column to split on.
        threshold : float
            The split threshold.

        Returns
        -------
        float
            The information gain.

        """
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # return the information gain
        return parent_entropy - child_entropy

    def _split(
        self,
        X_column: NDArray[np.float64],
        split_thresh: float,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        """Split the data based on the split threshold.

        Parameters
        ----------
        X_column : NDArray[np.float64]
            The input feature column.
        split_thresh : float
            The threshold value for the split.

        Returns
        -------
        tuple[NDArray[np.integer], NDArray[np.integer]]
            The indices of the left and right splits.

        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y: NDArray[np.integer]) -> float:
        """Return the entropy of y.

        formula: H(y) = -Σ(p * log(p)) where p is the relative frequency of each class.
        """
        hist = np.bincount(y)  # count the raw frequencies of each class
        ps = hist / len(y)  # calculate the relative frequencies
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y: NDArray[np.integer]) -> Literal[0, 1]:
        """Return the most common label in y: 0 or 1 (Binary Classification)."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.integer]:
        """Predict the class labels for the input data X.

        Parameters
        ----------
        X : NDArray[np.float64]
            The input data to predict.

        Returns
        -------
        NDArray[np.integer]
            The predicted class labels.

        """
        if self.root is not None:
            return np.array([self._traverse_tree(x, self.root) for x in X])
        msg = "The decision tree has not been fitted yet. Call fit() before predict()."
        raise AssertionError(msg)

    def _traverse_tree(self, x: NDArray[np.float64], node: Node) -> int:
        """Traverse the decision tree to predict the class label for a single sample x.

        Parameters
        ----------
        x : NDArray[np.float64]
            The input sample to predict.
        node : Node
            The current node in the decision tree.

        Returns
        -------
        int
            The predicted class label (0 or 1).

        """
        # if node.is_leaf_node():  # same thing but type checker doesn't understand it
        if node.value is not None:
            return node.value

        # if the node is not a leaf node (value is None), then it has a feature and threshold to split on
        if node.feature is None or node.threshold is None or node.left is None or node.right is None:
            msg = (
                "Something bad happened in the code. Non-leaf nodes should have feature, threshold, left, right set. "
                "Check method _grow_tree."
            )
            raise AssertionError(msg)

        # if it is not a leaf node: traverse tree
        if x[node.feature] <= node.threshold:
            # go left if the feature value is less than or equal to the threshold
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
