"""Implements a Gaussian Process Regression from scratch in pure python plus numpy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GaussianProcessRegression:
    """Gaussian Process Regression model."""

    def __init__(self, kernel: RBFKernel, sigma_n: float = 1e-8) -> None:
        """Initialize a Gaussian Process Regression model with a kernel and a variance."""
        self.kernel = kernel
        self.sigma_n = sigma_n

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit the Gaussian Process Regression model to the training data."""
        self.X_train = X
        self.y_train = y

        self.n = len(X)

        self.K = self.kernel(X, X) + np.eye(len(X)) * self.sigma_n  # Add noise term here
        self.K_inv = np.linalg.inv(self.K)

    def predict(
        self,
        X: NDArray[np.float64],
        *,
        diag: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict the mean and covariance of the Gaussian Process Regression model."""
        full_X = np.vstack((self.X_train, X))
        full_K = self.kernel(full_X, full_X)

        K_star = full_K[: self.n, self.n :]
        K_starstar = full_K[self.n :, self.n :]

        KstarTKinv = np.matmul(K_star.T, self.K_inv)
        mean = np.matmul(KstarTKinv, self.y_train)
        cov = K_starstar - np.matmul(KstarTKinv, K_star)

        return mean, np.diag(cov) if diag else cov


class RBFKernel:
    """RBF (Gaussian) Kernel for Gaussian Process Regression."""

    def __init__(self, length_scale: float = 1.0, sigma_scale: float = 1.0) -> None:
        """Initialize an RBF (Gaussian) Kernel for Gaussian Process Regression."""
        self.length_scale = length_scale
        self.sigma_scale = sigma_scale

    def __call__(self, X1: NDArray[np.float64], X2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the RBF kernel between two sets of points and return the kernel matrix."""
        euclidean_matrix = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1)

        return self.sigma_scale * np.exp(-0.5 * (euclidean_matrix / self.length_scale) ** 2)
