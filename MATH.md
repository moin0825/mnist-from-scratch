# The Math, in Full

This is the complete walkthrough of every computation in `network.py` — the forward pass, the loss, and every gradient in the backward pass, derived by hand.

---

## 1. Architecture and Shapes

```
Input (784)  -->  [W1, b1]  -->  ReLU  -->  Hidden (128)  -->  [W2, b2]  -->  Softmax  -->  Output (10)
```

Shapes (batch size = m):

| Symbol | Shape | Meaning |
|---|---|---|
| `X` | (m, 784) | input batch |
| `W1` | (784, 128) | input→hidden weights |
| `b1` | (1, 128) | hidden bias |
| `Z1` | (m, 128) | hidden pre-activation = X·W1 + b1 |
| `A1` | (m, 128) | hidden activation = ReLU(Z1) |
| `W2` | (128, 10) | hidden→output weights |
| `b2` | (1, 10) | output bias |
| `Z2` | (m, 10) | output pre-activation = A1·W2 + b2 |
| `A2` | (m, 10) | predicted probabilities = softmax(Z2) |

---

## 2. Forward Pass

```
Z1 = X · W1 + b1
A1 = ReLU(Z1) = max(0, Z1)

Z2 = A1 · W2 + b2
A2 = softmax(Z2)
```

**ReLU** zeroes out negative values. Cheap to compute, gradient is trivial (0 or 1) — the standard choice for hidden layers.

**Softmax** turns the 10 raw output scores into probabilities that sum to 1:

```
softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
```

In code we subtract the row-max before exponentiating (`z - max(z)`). This doesn't change the result — softmax is shift-invariant — but it stops `exp()` from overflowing on large inputs.

---

## 3. Loss: Cross-Entropy

For a single example with true one-hot label `y` and predicted probabilities `A2`:

```
L = - sum_i( y_i * log(A2_i) )
```

Since `y` is one-hot, this collapses to `-log(A2_correct_class)`. The loss only cares about the probability the model assigned to the *correct* digit. 99% probability on the right answer → loss near 0. 1% probability on the right answer → loss is large (`-log(0.01) ≈ 4.6`).

Averaged over the batch:

```
Loss = (1/m) * sum over batch of L
```

---

## 4. Backward Pass — Deriving Every Gradient

No autograd — the chain rule applied by hand, layer by layer, back to front.

### Step 1 — Gradient at the output layer (dL/dZ2)

When softmax is paired with cross-entropy loss, the combined derivative simplifies beautifully. Instead of separately computing dL/dA2 and dA2/dZ2 (a full Jacobian, since every output depends on every other output through the softmax normalization), the two steps collapse to:

```
dL/dZ2 = A2 - Y
```

Predicted probabilities minus the true one-hot label. If the model predicted 0.9 for the correct class (true value 1), the gradient is -0.1 (push it up further). If it predicted 0.9 for a wrong class (true value 0), the gradient is +0.9 (push it down hard).

*(Derivation: L = -sum_i y_i log(softmax(Z2)_i). Differentiating with respect to any single Z2_k, and using d(softmax_i)/d(Z2_k) = softmax_i(δ_ik − softmax_k), all cross-terms cancel, leaving dL/dZ2_k = A2_k − y_k.)*

### Step 2 — Gradients for W2 and b2

`Z2 = A1 · W2 + b2`, so:

```
dL/dW2 = A1^T · dZ2
dL/db2 = sum over batch of dZ2
```

`dL/dW2` tells you, for each hidden neuron feeding into each output neuron, how much that connection contributed to the error — the hidden activation that "sent" the signal, scaled by how wrong the output was.

### Step 3 — Propagate the error back into the hidden layer

Push the output-layer error backward through W2:

```
dL/dA1 = dZ2 · W2^T
```

Then through the ReLU: gradient passes unchanged where the input was positive, blocked (set to 0) where negative — a "dead" ReLU neuron had zero effect on the output, so it gets zero blame.

```
dL/dZ1 = dL/dA1 * ReLU'(Z1),   where ReLU'(z) = 1 if z > 0 else 0
```

(`*` here is element-wise, not matrix multiplication.)

### Step 4 — Gradients for W1 and b1

Same pattern, one layer earlier:

```
dL/dW1 = X^T · dZ1
dL/db1 = sum over batch of dZ1
```

### Summary — all four gradients

```
dZ2 = (A2 - Y) / m
dW2 = A1^T · dZ2
db2 = sum(dZ2, axis=0)

dA1 = dZ2 · W2^T
dZ1 = dA1 * ReLU'(Z1)
dW1 = X^T · dZ1
db1 = sum(dZ1, axis=0)
```

(The `/m` is folded into `dZ2` up front so every downstream gradient is already averaged over the batch.)

---

## 5. Optimizer: Plain Gradient Descent

No Adam, no momentum, no LR schedules:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

Applied per mini-batch (64 examples). Full-batch gradient descent is smooth but slow (one update per full pass over 60,000 images); single-example SGD is fast per step but noisy. Batch size 64 is the standard middle ground.

---

## 6. Training Setup

| Hyperparameter | Value |
|---|---|
| Hidden layer size | 128 |
| Epochs | 30 |
| Batch size | 64 |
| Learning rate | 0.5 |
| Weight init | He initialization (`N(0, sqrt(2/fan_in))`) |

**He initialization**: if weights start too small, activations shrink to zero across layers; too large, they explode. Scaling by `sqrt(2/fan_in)` keeps activation variance roughly stable at the start of training, specifically for ReLU networks.

---

## 7. Results and Overfitting

- Final test accuracy: 98.31% (9,831 / 10,000), target was 85%.
- Training loss: 0.25 (epoch 1) → 0.0007 (epoch 30).
- Per-digit accuracy is fairly even (97.8%–99.2%) — see `evaluate.py` output.
- Training loss keeps dropping toward ~0 in later epochs while test accuracy plateaus around epoch 15–20. That gap is mild overfitting — expected with no regularization (no dropout, no weight decay). Doesn't hurt the final result here, but it's why test accuracy, not training loss, is the number that matters.

---

## 8. Dataset Note

MNIST is loaded from a pickled, pre-flattened, pre-normalized `.pkl.gz` file in `data/`. `data_loader.py` merges the standard 50k/10k train/validation split into a single 60k training set and keeps the 10k test set separate, matching the classic MNIST 60k/10k convention.
