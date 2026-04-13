"""Implements a Support Vector Machine from scratch in pure python plus numpy.

This implementation is based on:
- https://www.youtube.com/watch?v=T9UcK-TxQGw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=10
- https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/blob/main/09%20SVM/svm.py

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SVM:
    """
    A simple linear Support Vector Machine (SVM) classifier.

    The goal is to find a boundary (line, plane, etc.) that separates two classes
    while maximizing the margin between them.

    This implementation uses:
    - hinge loss
    - L2 regularization
    - stochastic gradient descent (SGD)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_iters: int = 1000,
    ) -> None:
        """Initialize the SVM model.

        Parameters
        ----------
        learning_rate:
            Step size for updating the model parameters.

        lambda_param:
            Strength of regularization (controls how large weights can grow).

        n_iters:
            Number of passes over the dataset.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.w: NDArray[np.float64]
        self.b: float

    def fit(self, X: NDArray[np.float64], y: NDArray[np.integer]) -> None:
        """Train the SVM on the given data.

        The algorithm repeatedly:
        - checks whether each sample satisfies the margin condition
        - updates the weights and bias if it does not
        """
        n_samples, n_features = X.shape

        # Convert labels to -1 and +1 (required for SVM formulation)
        y_transformed = np.where(y <= 0, -1, 1)

        # Initialize weights randomly and bias to zero
        rng = np.random.default_rng()
        self.w = rng.normal(size=n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            # Shuffle samples each iteration (improves SGD behavior)
            for idx in rng.permutation(n_samples):
                x_i = X[idx]
                y_i = y_transformed[idx]

                # Check if sample satisfies margin condition:
                # y * (w · x + b) >= 1
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # If condition is satisfied:
                    # only apply regularization (shrink weights slightly)
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # If condition is violated:
                    # update weights to better classify this sample
                    # and increase margin
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - y_i * x_i)

                    # Update bias (not regularized)
                    self.b += self.learning_rate * y_i

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.integer]:
        """
        Predict class labels for each sample in X.

        A sample is classified based on which side of the boundary it lies on.
        """
        # Compute decision scores
        scores = np.dot(X, self.w) + self.b

        # Convert to class labels (0 or 1)
        return np.where(scores >= 0, 1, 0)
