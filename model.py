"""
Bayesian and Robust Logistic Regression from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - sigmoid
import numpy as np

def sigmoid(a):
    # Convert the input to a floating-point NumPy array.
    a = np.asarray(a, dtype=float)

    # Allocate an output array with the same shape as the input.
    result = np.empty_like(a, dtype=float)

    # For non-negative logits, evaluate 1 / (1 + exp(-a)).
    # The exponent is never positive, preventing overflow.
    nonnegative = a >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-a[nonnegative]))

    # For negative logits, use the equivalent expression
    # exp(a) / (1 + exp(a)), again avoiding positive exponents.
    negative = ~nonnegative
    exp_a = np.exp(a[negative])
    result[negative] = exp_a / (1.0 + exp_a)

    return result

# Step 2 - logreg_nll
def logreg_nll(w, X, y):
    # Compute the logits for all examples.
    a = X @ w

    # Compute binary cross-entropy directly from logits.
    # np.logaddexp(0.0, a) evaluates log(1 + exp(a))
    # stably, avoiding overflow for extreme logits.
    nll = np.logaddexp(0.0, a) - y * a

    # Return the average negative log-likelihood as a Python float.
    return float(np.mean(nll))

