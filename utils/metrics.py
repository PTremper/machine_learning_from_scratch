import numpy as np


def accuracy(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy of the predictions.

    accuracy = (true positives + true negatives) / total
    """
    return np.sum(y_test == y_pred) / len(y_test)
