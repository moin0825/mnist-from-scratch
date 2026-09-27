"""
evaluate.py
-----------
Loads saved trained parameters and evaluates them on the MNIST test set.
Run this after train.py to re-check accuracy without retraining.
"""

import numpy as np
from data_loader import load_mnist
from network import predict


def load_params(path="trained_params.npz"):
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def per_class_accuracy(preds, labels, num_classes=10):
    """Returns a dict {digit: accuracy} so you can see which digits are hardest."""
    results = {}
    for c in range(num_classes):
        mask = labels == c
        results[c] = np.mean(preds[mask] == labels[mask])
    return results


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, y_test_labels = load_mnist()
    params = load_params("trained_params.npz")

    preds = predict(params, X_test)
    overall_acc = np.mean(preds == y_test_labels)
    print(f"Overall test accuracy: {overall_acc*100:.2f}%  ({int(overall_acc*len(y_test_labels))}/{len(y_test_labels)} correct)\n")

    print("Per-digit accuracy:")
    for digit, acc in per_class_accuracy(preds, y_test_labels).items():
        print(f"  {digit}: {acc*100:.2f}%")
