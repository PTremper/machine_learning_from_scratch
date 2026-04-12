"""Implements a Support Vector Machine from scratch in pure python plus numpy.

This implementation is based on:
- https://www.youtube.com/watch?v=T9UcK-TxQGw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=10
- https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/blob/main/09%20SVM/svm.py

"""

import numpy as np
from numpy.typing import NDArray


class SVM:
    """A simple support vector machine.

    Methods
    -------
    fit(X, y)
        Fit the SVM to the training data.
    predict(X)
        Predict the class labels for the input data.

    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_iters: int = 1000,
    ) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.w: NDArray[np.float64]
        self.b: float

    def fit(self, X: NDArray[np.float64], y: NDArray[np.integer]) -> None:
        """Fits the SVM model to the training data."""
        _n_samples, n_features = X.shape

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # init weights
        rng = np.random.default_rng()
        self.w = rng.normal(size=n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.integer]:
        """Predicts the class labels for the given input data."""
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
