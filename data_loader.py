"""
data_loader.py
--------------
Loads the MNIST dataset and prepares it for training.

The raw file (data/mnist.pkl.gz) stores images as flattened 784-length
float32 vectors already scaled to [0, 1], split into three parts:
train (50,000), validation (10,000), test (10,000).

We merge train + validation into a single 60,000-image training set,
matching the standard MNIST 60k/10k split, and one-hot encode the labels
since the network's output layer is 10 softmax neurons, not a single
class index.
"""

import gzip
import pickle
import numpy as np


def one_hot_encode(labels, num_classes=10):
    """Convert an array of integer labels (N,) into one-hot vectors (N, num_classes)."""
    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def load_mnist(path="data/mnist.pkl.gz"):
    """
    Returns:
        X_train: (60000, 784) float array, pixel values in [0, 1]
        y_train: (60000, 10) one-hot labels
        X_test:  (10000, 784) float array, pixel values in [0, 1]
        y_test:  (10000, 10) one-hot labels
        y_test_labels: (10000,) raw integer labels, kept for readable accuracy reporting
    """
    import warnings
    with gzip.open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_set, val_set, test_set = pickle.load(f, encoding="latin1")

    X_train = np.concatenate([train_set[0], val_set[0]], axis=0).astype(np.float64)
    y_train_labels = np.concatenate([train_set[1], val_set[1]], axis=0)

    X_test = test_set[0].astype(np.float64)
    y_test_labels = test_set[1]

    y_train = one_hot_encode(y_train_labels)
    y_test = one_hot_encode(y_test_labels)

    return X_train, y_train, X_test, y_test, y_test_labels


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, y_test_labels = load_mnist()
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("pixel range:", X_train.min(), "-", X_train.max())
