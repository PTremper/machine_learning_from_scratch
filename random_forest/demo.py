"""Demo of the Random Forest classifier on a benchmark dataset."""

from machine_learning_from_scratch.random_forest.random_forest import RandomForest
from machine_learning_from_scratch.utils.metrics import accuracy
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main() -> None:
    """Demo of the Random Forest classifier on the breast cancer dataset."""
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1234,
    )

    clf = RandomForest(n_trees=20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = accuracy(y_test, predictions)

    print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
