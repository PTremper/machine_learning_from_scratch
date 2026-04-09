"""Implements a Neural Network from scratch in pure python plus numpy."""

from __future__ import annotations

import numpy as np


class NeuralNetwork:
    """A neural network composed of multiple dense layers."""

    def __init__(self) -> None:
        """Initialize an empty neural network with no layers."""
        self.layers = []  # Initialize an empty list to hold the layers of the network

    def add_layer(self, layer: DenseLayer) -> None:
        """Add a layer to the network."""
        self.layers.append(
            layer,
        )  # Add a layer to the network by appending it to the list of layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform a forward pass through the network."""
        for layer in self.layers:
            x = layer.forward(
                x,
            )  # Perform a forward pass through each layer of the network
        return x  # Return the final output of the forward pass

    def backward(self, loss_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Perform backpropagation through the network."""
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(
                loss_gradient,
                learning_rate,
            )  # Perform backpropagation through each layer
        return loss_gradient  # Return the final loss gradient for external use


class Activation:
    """Base activation function with trivial behaviour."""

    def __init__(self) -> None:
        """Initialize the activation function."""

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the activation function."""
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the activation function."""
        return x


class ReLu(Activation):
    """ReLu activation function."""

    def __init__(self) -> None:
        """Initialize the ReLu activation function."""
        super().__init__()

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the ReLu activation function."""
        return (x > 0).astype(x.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the ReLu activation function."""
        return np.maximum(0, x)


class Sigmoid(Activation):
    """Sigmoid activation function."""

    def __init__(self) -> None:
        """Initialize the sigmoid activation function."""
        super().__init__()

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid activation function."""
        sigmoid_output = self.forward(x)
        return sigmoid_output * (1 - sigmoid_output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))


class ReLuSigmoid(Activation):
    """ReLuSigmoid activation function."""

    def __init__(self) -> None:
        """Initialize the ReLuSigmoid activation function."""
        super().__init__()
        self.relu = ReLu()
        self.sigmoid = Sigmoid()

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the ReLuSigmoid activation function."""
        return self.relu.derivative(self.sigmoid.forward(x)) * self.sigmoid.derivative(
            x,
        )  # chain rule

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the ReLuSigmoid activation function."""
        return self.relu.forward(self.sigmoid.forward(x))


class DenseLayer:
    """A dense (fully connected) layer for a neural network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Activation | None = None,
    ) -> None:
        """Initialize the dense layer with random weights and zeros biases."""
        rng = np.random.default_rng()
        self.weights: np.ndarray = rng.standard_normal((input_size, output_size))
        self.bias: np.ndarray = np.zeros((1, output_size))
        self.input: np.ndarray  # Store the input data for use during backpropagation
        self.output: np.ndarray  # Store the output data for use during backpropagation

        if activation is None:
            self.activation = Activation()
        else:
            self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward pass of the dense layer."""
        # Ensure input is a 2D array: pads shape if not a batch
        self.input = np.array(x, ndmin=2)

        # Calculate the weighted sum and add biases
        weighted_sum = np.dot(self.input, self.weights) + self.bias

        # Apply activation function
        self.output = self.activation.forward(weighted_sum)

        return self.output

    def backward(self, loss_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Compute the backward pass of the dense layer."""
        # Apply the derivative of the activation function
        loss_gradient *= self.activation.derivative(self.output)

        # Compute gradients for weights, biases, and input
        weights_gradient = np.dot(self.input.T, loss_gradient)
        bias_gradient = np.sum(loss_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(loss_gradient, self.weights.T)

        # Update weights and biases using gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient  # Return the gradient to be passed to the previous layer
