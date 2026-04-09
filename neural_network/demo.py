"""Demo of the Neural Network on a simple artificial dataset."""  # noqa: INP001

import numpy as np
from machine_learning_from_scratch.neural_network.neural_network import DenseLayer, NeuralNetwork, ReLu, Sigmoid
from tqdm import trange


def main() -> None:
    """Train and test a neural network on a simple artificial dataset."""
    # Example usage:
    # Create a neural network
    nn = NeuralNetwork()

    # Add layers to the neural network
    nn.add_layer(DenseLayer(2, 8, activation=ReLu()))
    nn.add_layer(DenseLayer(8, 1, activation=Sigmoid()))

    # Define some input data and target output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Training loop
    learning_rate = 0.1
    epochs = 10000

    tqdm_range = trange(epochs)

    for epoch in tqdm_range:
        for i in range(len(X)):
            # Forward pass
            output = nn.forward(X[i])

            # Calculate loss (MSE)
            loss = 0.5 * np.sum((output - y[i]) ** 2)

            # Backpropagation
            loss_gradient = output - y[i]
            nn.backward(loss_gradient, learning_rate)

        tqdm_range.set_description(
            f"[Epoch {epoch + 1}/{epochs}] Loss: {loss:.3f}",
            refresh=True,
        )

    # Test the trained network
    for i in range(len(X)):
        output = nn.forward(X[i])
        print(f"Input: {X[i]}, Predicted Output: {output}, Label: {y[i]}")  # noqa: T201


if __name__ == "__main__":
    main()
