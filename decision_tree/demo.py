"""Demo of the Decision Tree classifier on a benchmark dataset."""

import numpy as np
from machine_learning_from_scratch.decision_tree.decision_tree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234,
)

clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


def accuracy(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy of the predictions.

    accuracy = (true positives + true negatives) / total
    """
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)

print(f"Accuracy: {acc:.3f}")  # noqa: T201
