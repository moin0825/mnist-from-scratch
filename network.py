"""
network.py
----------
A 2-layer (1 hidden layer) feedforward neural network, implemented with
NumPy only. Every piece of math — forward pass, loss, gradients, weight
update — is written out by hand. No autograd, no framework.

Architecture:
    input (784) -> W1, b1 -> ReLU -> hidden (128) -> W2, b2 -> softmax -> output (10)

See README.md for the derivation of every gradient used below.
"""

import numpy as np


def init_params(input_size=784, hidden_size=128, output_size=10, seed=42):
    """
    He initialization for ReLU layers: weights ~ N(0, 2/fan_in).
    Keeps activations from exploding or vanishing at the start of training.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ---------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(z.dtype)


def softmax(z):
    """
    Row-wise softmax with the max-subtraction trick for numerical stability
    (subtracting the row max doesn't change the result, since softmax is
    shift-invariant, but stops exp() from overflowing).
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

def forward(params, X):
    """
    Runs one forward pass.

    X: (batch, 784)
    Returns a cache of every intermediate value, needed later for backprop.
    """
    Z1 = X @ params["W1"] + params["b1"]      # (batch, 128) pre-activation
    A1 = relu(Z1)                              # (batch, 128) hidden activation
    Z2 = A1 @ params["W2"] + params["b2"]      # (batch, 10) pre-activation
    A2 = softmax(Z2)                           # (batch, 10) predicted probabilities

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

def cross_entropy_loss(A2, Y):
    """
    Mean categorical cross-entropy over a batch.
    A2: (batch, 10) predicted probabilities (softmax output)
    Y:  (batch, 10) one-hot true labels
    """
    m = Y.shape[0]
    eps = 1e-12  # avoid log(0)
    log_probs = np.log(np.clip(A2, eps, 1.0))
    loss = -np.sum(Y * log_probs) / m
    return loss


# ---------------------------------------------------------------------
# Backward pass (manual chain rule)
# ---------------------------------------------------------------------

def backward(params, cache, Y):
    """
    Computes dLoss/dW1, dLoss/db1, dLoss/dW2, dLoss/db2.

    Key simplification: for softmax + cross-entropy together, the gradient
    of the loss w.r.t. the output layer's pre-activation Z2 collapses to
    a clean (A2 - Y). Full derivation is in README.md.
    """
    X, A1, Z1 = cache["X"], cache["A1"], cache["Z1"]
    A2 = cache["A2"]
    m = Y.shape[0]

    # ---- Output layer ----
    dZ2 = (A2 - Y) / m                  # (batch, 10)
    dW2 = A1.T @ dZ2                    # (128, 10)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, 10)

    # ---- Hidden layer ----
    dA1 = dZ2 @ params["W2"].T          # (batch, 128)
    dZ1 = dA1 * relu_derivative(Z1)     # (batch, 128), chain rule through ReLU
    dW1 = X.T @ dZ1                     # (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, 128)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# ---------------------------------------------------------------------
# Optimizer: plain (mini-batch) gradient descent
# ---------------------------------------------------------------------

def sgd_update(params, grads, learning_rate):
    """Vanilla gradient descent step: theta = theta - lr * dtheta. No momentum, no Adam."""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


def predict(params, X):
    """Returns predicted class indices (batch,) for a batch of inputs."""
    A2, _ = forward(params, X)
    return np.argmax(A2, axis=1)
