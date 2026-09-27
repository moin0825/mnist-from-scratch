"""
train.py
--------
Mini-batch gradient descent training loop for the from-scratch network.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_mnist
from network import init_params, forward, cross_entropy_loss, backward, sgd_update, predict


def iterate_minibatches(X, Y, batch_size, rng):
    """Yields shuffled mini-batches of (X, Y) for one epoch."""
    n = X.shape[0]
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], Y[batch_idx]


def evaluate_accuracy(params, X, y_labels):
    """Fraction of correctly classified examples. y_labels is integer (not one-hot)."""
    preds = predict(params, X)
    return np.mean(preds == y_labels)


def train(
    X_train, y_train, X_test, y_test, y_test_labels,
    hidden_size=128,
    epochs=30,
    batch_size=64,
    learning_rate=0.5,
    seed=42,
):
    """
    Trains the network with mini-batch SGD and tracks:
      - average training loss per epoch
      - test accuracy per epoch (for visibility into progress)

    Returns the trained params and a history dict.
    """
    rng = np.random.default_rng(seed)
    params = init_params(input_size=X_train.shape[1], hidden_size=hidden_size, seed=seed)

    history = {"train_loss": [], "test_accuracy": []}

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for X_batch, Y_batch in iterate_minibatches(X_train, y_train, batch_size, rng):
            A2, cache = forward(params, X_batch)
            loss = cross_entropy_loss(A2, Y_batch)
            grads = backward(params, cache, Y_batch)
            params = sgd_update(params, grads, learning_rate)
            epoch_losses.append(loss)

        avg_loss = float(np.mean(epoch_losses))
        test_acc = evaluate_accuracy(params, X_test, y_test_labels)

        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(test_acc)

        print(f"Epoch {epoch:2d}/{epochs} | train loss: {avg_loss:.4f} | test accuracy: {test_acc*100:.2f}%")

    return params, history


def plot_loss_curve(history, out_path="loss_curve.png"):
    """Saves a plot of training loss vs. epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, history["train_loss"], color="tab:red", marker="o", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [a * 100 for a in history["test_accuracy"]], color="tab:blue", marker="s", label="Test accuracy")
    ax2.set_ylabel("Test accuracy (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Training Loss and Test Accuracy over Epochs")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {out_path}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, y_test_labels = load_mnist()

    params, history = train(
        X_train, y_train, X_test, y_test, y_test_labels,
        hidden_size=128,
        epochs=30,
        batch_size=64,
        learning_rate=0.5,
    )

    plot_loss_curve(history)

    final_acc = evaluate_accuracy(params, X_test, y_test_labels)
    print(f"\nFinal test accuracy: {final_acc*100:.2f}%")

    np.savez("trained_params.npz", **params)
    print("Saved trained parameters to trained_params.npz")
