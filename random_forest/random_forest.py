"""Implements a Random Forest from scratch in pure python plus numpy.

This implementation is based on:
- https://www.youtube.com/watch?v=kFwe2ZZU7yw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=6
- https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/blob/main/05%20Random%20Forests/RandomForest.py

"""

from collections import Counter

import numpy as np
from machine_learning_from_scratch.decision_tree.decision_tree import DecisionTree
from numpy.typing import NDArray
from tqdm import tqdm


class RandomForest:
    """A simple random forest classifier.

    The idea is straightforward:
    - train many decision trees
    - each tree sees a slightly different version of the data
    - combine their predictions (majority vote)

    This reduces overfitting and makes predictions more robust.
    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: int | None = None,
    ) -> None:
        """Initialize the random forest model.

        Parameters
        ----------
        n_trees:
            Number of decision trees in the forest.

        max_depth:
            Maximum depth of each tree.

        min_samples_split:
            Minimum number of samples required to keep splitting in each tree.

        max_features:
            Number of features (dimensions) each tree considers when splitting.

        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.max_features = max_features
        self.trees: list[DecisionTree] = []

        self.rng = np.random.default_rng()

    def fit(self, X: NDArray[np.float64], y: NDArray[np.integer]) -> None:
        """Train the random forest.

        Each tree is trained on a resampled (bootstrapped) version of the dataset.
        """
        self.trees = []

        for _ in tqdm(range(self.n_trees)):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_sample_split,
                max_features=self.max_features,
            )

            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.integer],
    ) -> tuple[NDArray[np.float64], NDArray[np.integer]]:
        """Create a new dataset by sampling with replacement.

        - same number of samples as original
        - some samples may appear multiple times
        - some samples may be left out
        """
        n_samples = X.shape[0]

        sample_indices = self.rng.choice(
            n_samples,
            n_samples,
            replace=True,  # samples arent being dropped from the pool
        )
        return X[sample_indices], y[sample_indices]

    def _most_common_label(self, labels: NDArray[np.integer]) -> int:
        """Return the most frequent label."""
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.integer]:
        """Predict class labels for each sample in X.

        Each tree makes a prediction, and the final result is the majority vote.
        """
        # shape: (n_trees, n_samples)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # shape: (n_samples, n_trees)
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)

        # majority vote per sample over all trees
        final_predictions = np.array(
            [self._most_common_label(preds) for preds in tree_predictions],
        )

        return final_predictions  # noqa: RET504
