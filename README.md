# MNIST Digit Classifier — Built from Scratch with NumPy

A feedforward neural network that classifies handwritten digits (0–9),
built using only NumPy. No PyTorch, TensorFlow, or scikit-learn for the
model — every forward pass, loss, gradient, and weight update is
written out by hand.

**Result: 98.3% test accuracy** (target was 85%+).

---

## 1. Files

| File | What it does |
|---|---|
| `data_loader.py` | Loads MNIST, merges to 60k train / 10k test, normalizes, one-hot encodes labels |
| `network.py` | Parameter init, forward pass, loss, backward pass (manual gradients), SGD update |
| `train.py` | Mini-batch training loop, tracks loss/accuracy per epoch, saves loss curve |
| `evaluate.py` | Loads a trained model and reports test accuracy (overall + per-digit) |
| `loss_curve.png` | Plot of training loss and test accuracy over 30 epochs |
| `data/mnist.pkl.gz` | The dataset (images already flattened to 784-length vectors, pixels in [0,1]) |

To run everything from scratch:

```bash
python3 train.py      # trains for 30 epochs, saves trained_params.npz and loss_curve.png
python3 evaluate.py    # re-checks accuracy on the test set using the saved weights
```

---

## 2. The Architecture

```
Input (784)  -->  [W1, b1]  -->  ReLU  -->  Hidden (128)  -->  [W2, b2]  -->  Softmax  -->  Output (10)
```

Each 28×28 image is flattened into a 784-length vector. That vector is
multiplied by a weight matrix, shifted by a bias, passed through a
non-linearity (ReLU), then multiplied by a second weight matrix, shifted
by a second bias, and turned into a probability distribution over the
10 digit classes (softmax).

**Shapes** (batch size = m):

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

## 3. Forward Pass

```
Z1 = X · W1 + b1
A1 = ReLU(Z1) = max(0, Z1)

Z2 = A1 · W2 + b2
A2 = softmax(Z2)
```

**ReLU** just zeroes out negative values. It's cheap to compute and its
gradient is trivial (0 or 1), which is why it's the standard choice for
hidden layers.

**Softmax** turns the 10 raw output scores into probabilities that sum
to 1:

```
softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
```

In code we subtract the row-max before exponentiating
(`z - max(z)`). This doesn't change the result — softmax is shift
invariant — but it stops `exp()` from overflowing on large inputs.

---

## 4. Loss: Cross-Entropy

For a single example with true one-hot label `y` and predicted
probabilities `A2`:

```
L = - sum_i( y_i * log(A2_i) )
```

Since `y` is one-hot (a single 1, rest 0s), this collapses to just
`-log(A2_correct_class)`. In other words: the loss only cares about
the probability the model assigned to the *correct* digit. If the
model puts 99% probability on the right answer, loss is close to 0.
If it puts 1% probability on the right answer, loss is large
(`-log(0.01) ≈ 4.6`).

We average this over the batch:

```
Loss = (1/m) * sum over batch of L
```

---

## 5. Backward Pass — Deriving Every Gradient

This is the core of "from scratch": no autograd, just the chain rule
applied by hand, layer by layer, back to front.

### Step 1 — Gradient at the output layer (dL/dZ2)

This is the one shortcut worth knowing well: when softmax is paired
with cross-entropy loss, the combined derivative simplifies
beautifully. Instead of separately computing dL/dA2 and dA2/dZ2 (which
involves a full Jacobian, since every output depends on every other
output through the softmax normalization), the two steps collapse to:

```
dL/dZ2 = A2 - Y
```

That's it — predicted probabilities minus the true one-hot label. This
makes intuitive sense: if the model predicted 0.9 for the correct
class (true value 1), the gradient is -0.1 (push it up further). If it
predicted 0.9 for a wrong class (true value 0), the gradient is +0.9
(push it down hard).

*(Short derivation: L = -sum_i y_i log(softmax(Z2)_i). Differentiating
this sum with respect to any single Z2_k, and using the fact that
d(softmax_i)/d(Z2_k) = softmax_i(δ_ik − softmax_k), all the
cross-terms cancel and you're left with dL/dZ2_k = A2_k − y_k.)*

### Step 2 — Gradients for W2 and b2

`Z2 = A1 · W2 + b2`, so by the standard rules of matrix calculus:

```
dL/dW2 = A1^T · dZ2
dL/db2 = sum over batch of dZ2
```

Intuition: `dL/dW2` tells you, for each hidden neuron feeding into each
output neuron, how much that specific connection contributed to the
error — it's the hidden activation that "sent" the signal, scaled by
how wrong the output was.

### Step 3 — Propagate the error back into the hidden layer

First, push the output-layer error backward through W2 to find out how
much each hidden neuron contributed to the final error:

```
dL/dA1 = dZ2 · W2^T
```

Then push it through the ReLU: ReLU passes gradient through unchanged
where the input was positive, and blocks it (sets to 0) where the
input was negative or zero — because a "dead" ReLU neuron (output 0)
had zero effect on the output, so it gets zero blame.

```
dL/dZ1 = dL/dA1 * ReLU'(Z1),   where ReLU'(z) = 1 if z > 0 else 0
```

(`*` here is element-wise multiplication, not matrix multiplication.)

### Step 4 — Gradients for W1 and b1

Same pattern as Step 2, one layer earlier:

```
dL/dW1 = X^T · dZ1
dL/db1 = sum over batch of dZ1
```

### Summary of the four gradients we need

```
dZ2 = (A2 - Y) / m
dW2 = A1^T · dZ2
db2 = sum(dZ2, axis=0)

dA1 = dZ2 · W2^T
dZ1 = dA1 * ReLU'(Z1)
dW1 = X^T · dZ1
db1 = sum(dZ1, axis=0)
```

(The `/m` is folded into `dZ2` up front so every downstream gradient
is already averaged over the batch — one division instead of four.)

---

## 6. Optimizer: Plain Gradient Descent

No Adam, no momentum, no learning rate schedules. Just:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

Applied per mini-batch (mini-batch SGD), not per single example or
over the whole dataset at once. Mini-batches (64 examples here) are
the standard middle ground: full-batch gradient descent is smooth but
slow (one update per full pass over 60,000 images), while single-
example SGD is fast per step but noisy. Batches of 64 strike a
reasonable balance and are what let training converge in a handful of
epochs.

---

## 7. Training Setup

| Hyperparameter | Value |
|---|---|
| Hidden layer size | 128 |
| Epochs | 30 |
| Batch size | 64 |
| Learning rate | 0.5 |
| Weight init | He initialization (`N(0, sqrt(2/fan_in))`), suited to ReLU |

**He initialization** matters more than it might look: if weights start
too small, activations shrink to zero as they pass through layers; too
large, and they explode. Scaling the initial random weights by
`sqrt(2/fan_in)` keeps the variance of activations roughly stable
across layers at the start of training, specifically for ReLU networks.

---

## 8. Results

- **Final test accuracy: 98.31%** (9,831 / 10,000 correct), well above
  the 85% target.
- Training loss dropped from 0.25 (epoch 1) to 0.0007 (epoch 30).
- Per-digit accuracy is fairly even (97.8%–99.2%), no digit the model
  is dramatically worse at. See `evaluate.py` output for the exact
  breakdown.
- See `loss_curve.png` for training loss and test accuracy plotted over
  all 30 epochs. Loss drops fast in the first 5 epochs, then flattens;
  test accuracy tracks it, with a small amount of noise batch-to-batch.

Note on overfitting: training loss keeps dropping toward ~0 in later
epochs while test accuracy plateaus around epoch 15–20. That gap is
mild overfitting — expected for a model with no regularization
(no dropout, no weight decay) trained for 30 epochs. It doesn't hurt
the final result here, but it's why test accuracy, not training loss,
is the number that matters.

---

## 9. Dataset Note

MNIST is loaded from a pickled, pre-flattened, pre-normalized `.pkl.gz`
file in `data/`. `data_loader.py` merges the standard 50k/10k
train/validation split into a single 60k training set and keeps the
10k test set separate, matching the classic MNIST 60k/10k convention.
