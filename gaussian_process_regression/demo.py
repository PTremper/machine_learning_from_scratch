"""
Minimal demonstration of Gaussian Process Regression.

This script shows how predictions can be made using a Gaussian
process with a predefined kernel, without focusing on training
or hyperparameter optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
from machine_learning_from_scratch.gaussian_process_regression.gaussian_process_regression import (
    GaussianProcessRegression,
    RBFKernel,
)
from machine_learning_from_scratch.utils.show_or_save import show_or_save


def main() -> None:
    """
    Demonstrate Gaussian Process regression on a simple dataset derived from a sine function.

    The example computes predictions using a predefined kernel
    and visualizes or prints the resulting mean estimates,
    without performing hyperparameter training.
    """
    rng = np.random.default_rng()

    def true_function(x: np.ndarray) -> np.ndarray:
        return x * np.sin(x) + 2 * x

    # generate data
    x_min, x_max, x_steps = [0, 10, 11]

    X = (np.linspace(x_min, x_max, x_steps) + rng.normal(0, (x_max - x_min) / x_steps, x_steps)).reshape(-1, 1)
    y = true_function(X.reshape(-1))

    # initialize kernel and GP
    kernel = RBFKernel()
    gpr = GaussianProcessRegression(kernel=kernel)

    # Fit GP to the support points
    gpr.fit(X=X, y=y)

    # Prediction
    x_axis = np.linspace(x_min, x_max, x_steps * 10)
    mean, cov = gpr.predict(x_axis.reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(16, 9))
    plt.title(
        f"Gaussian Process Regression\n{r'$\ell$'}"
        f"={gpr.kernel.length_scale:.2f}, {r'$\sigma$'}"
        f"={gpr.kernel.sigma_scale:.2f} (no training)",
    )
    plt.grid()
    plt.plot(x_axis, mean, color="blue", label="GPR Prediction")
    plt.fill_between(x=x_axis, y1=mean + np.sqrt(cov), y2=mean - np.sqrt(cov), color="orange", alpha=0.8, label="1 std")
    plt.plot(x_axis, true_function(x_axis), color="black", linestyle="dashed", label="True Function")
    plt.scatter(X.reshape(-1), y, c="red", label="Data Points")
    plt.legend()

    show_or_save("machine_learning_from_scratch/gaussian_process_regression/gaussian_process_regression_demo.png")


if __name__ == "__main__":
    main()
