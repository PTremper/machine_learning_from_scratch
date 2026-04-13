"""Demo of the Support Vector Machine classifier on a benchmark dataset.

This implementation is based on:
- https://www.youtube.com/watch?v=T9UcK-TxQGw&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=10
- https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/blob/main/09%20SVM/svm.py

"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from machine_learning_from_scratch.support_vector_machine.support_vector_machine import SVM
from machine_learning_from_scratch.utils.metrics import accuracy
from machine_learning_from_scratch.utils.show_or_save import show_or_save
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.model_selection import train_test_split


def get_decision_boundary_y(  # noqa: D103
    x: float,
    w: NDArray[np.float64],
    b: float,
    offset: float,
) -> float:
    return (-w[0] * x + b + offset) / w[1]


def main() -> None:
    """Support Vector Machine Demo."""
    parser = argparse.ArgumentParser(description="Support Vector Machine Demo")
    parser.add_argument("--n_iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--random_state", type=int, help="Random state")
    args = parser.parse_args()

    X, y = datasets.make_blobs(
        n_samples=50,
        n_features=2,
        centers=2,
        cluster_std=1.05,
        random_state=args.random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = SVM(n_iters=args.n_iters)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(f"SVM classification accuracy: {accuracy(y_test, predictions)}")

    # Visualisation, see
    # https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/blob/main/09%20SVM/svm.py
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_decision_boundary_y(x0_1, clf.w, clf.b, 0)
    x1_2 = get_decision_boundary_y(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_decision_boundary_y(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_decision_boundary_y(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_decision_boundary_y(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_decision_boundary_y(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim((x1_min - 3, x1_max + 3))

    show_or_save(
        "machine_learning_from_scratch/support_vector_machine/support_vector_machine_demo.png",
    )


if __name__ == "__main__":
    main()
