"""
Bayesian and Robust Logistic Regression from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - sigmoid
import numpy as np

def sigmoid(a):
    # Convert the input to a float array so lists, scalars, and integer arrays work.
    a = np.asarray(a, dtype=float)

    # Allocate an output array with the same shape as the input.
    result = np.empty_like(a, dtype=float)

    # For non-negative values, use 1 / (1 + exp(-a)).
    # The exponent -a is non-positive, preventing overflow.
    positive_mask = a >= 0
    result[positive_mask] = 1.0 / (
        1.0 + np.exp(-a[positive_mask])
    )

    # For negative values, use exp(a) / (1 + exp(a)).
    # The exponent a is negative, preventing overflow.
    negative_mask = ~positive_mask
    exp_a = np.exp(a[negative_mask])
    result[negative_mask] = exp_a / (1.0 + exp_a)

    return result

# Step 2 - logreg_nll
def logreg_nll(w, X, y):
    # Compute the logits: a = X @ w.
    a = X @ w

    # Compute the binary logistic regression NLL using the
    # numerically stable softplus: log(1 + exp(a)) - y * a.
    # np.logaddexp(0, a) avoids overflow for large positive logits.
    nll = np.logaddexp(0.0, a) - y * a

    # Return the average negative log-likelihood as a Python float.
    return float(np.mean(nll))

# Step 3 - logreg_grad
def logreg_grad(w, X, y):
    # Get the number of training examples.
    N = X.shape[0]
    # Compute predicted probabilities using sigmoid from Step 1.
    mu = sigmoid(X @ w)
    # Compute the gradient of the average negative log-likelihood.
    # The gradient is X.T @ (mu - y) / N.
    grad = X.T @ (mu - y) / N
    # Return the gradient vector of shape (D,).
    return grad

