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
    # Compute the logits for all examples.
    logits = X @ w

    # Convert logits to probabilities using the sigmoid function from Step 1.
    mu = sigmoid(logits)

    # Compute the average gradient of the negative log-likelihood.
    # X.T has shape (D, N), and (mu - y) has shape (N,).
    grad = X.T @ (mu - y) / X.shape[0]

    return grad

# Step 4 - logreg_hessian
def logreg_hessian(w, X):
    # Compute the logits for all examples.
    logits = X @ w

    # Compute predicted probabilities using the sigmoid function from Step 1.
    mu = sigmoid(logits)

    # Compute the diagonal weights: mu_i * (1 - mu_i).
    weights = mu * (1.0 - mu)

    # Scale the columns of X.T by the corresponding weights.
    # This avoids constructing an N x N diagonal matrix.
    weighted_X_T = X.T * weights

    # Compute the Hessian of the average negative log-likelihood.
    hessian = (weighted_X_T @ X) / X.shape[0]

    return hessian

# Step 5 - numerical_gradient
def numerical_gradient(f, w, eps=1e-5):
    # Create a floating-point copy so the original w is never modified.
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w, dtype=float)

    # Compute the gradient one coordinate at a time using central differences.
    for d in range(w.size):
        # Create independent perturbed copies of w.
        w_plus = w.copy()
        w_minus = w.copy()

        w_plus[d] += eps
        w_minus[d] -= eps

        # Estimate the partial derivative with respect to coordinate d.
        grad[d] = (f(w_plus) - f(w_minus)) / (2.0 * eps)

    return grad

# Step 6 - load_iris_sepal
def load_iris_sepal():
    # Store the sepal lengths locally so the function works even when
    # the global constants are unavailable in the test environment.
    setosa_data = (
        "5.1 4.9 4.7 4.6 5.0 5.4 4.6 5.0 4.4 4.9 5.4 4.8 4.8 4.3 5.8 "
        "5.7 5.4 5.1 5.7 5.1 5.4 5.1 4.6 5.1 4.8 5.0 5.0 5.2 5.2 4.7 "
        "4.8 5.4 5.2 5.5 4.9 5.0 5.5 4.9 4.4 5.1 5.0 4.5 4.4 5.0 5.1 "
        "4.8 5.1 4.6 5.3 5.0"
    )

    versicolor_data = (
        "7.0 6.4 6.9 5.5 6.5 5.7 6.3 4.9 6.6 5.2 5.0 5.9 6.0 6.1 5.6 "
        "6.7 5.6 5.8 6.2 5.6 5.9 6.1 6.3 6.1 6.4 6.6 6.8 6.7 6.0 5.7 "
        "5.5 5.5 5.8 6.0 5.4 6.0 6.7 6.3 5.6 5.5 5.5 6.1 5.8 5.0 5.6 "
        "5.7 5.7 6.2 5.1 5.7"
    )

    virginica_data = (
        "6.3 5.8 7.1 6.3 6.5 7.6 4.9 7.3 6.7 7.2 6.5 6.4 6.8 5.7 5.8 "
        "6.4 6.5 7.7 7.7 6.0 6.9 5.6 7.7 6.3 6.7 7.2 6.2 6.1 6.4 7.2 "
        "7.4 7.9 6.4 6.3 6.1 7.7 6.3 6.4 6.0 6.9 6.7 6.9 5.8 6.8 6.7 "
        "6.7 6.3 6.5 6.2 5.9"
    )

    # Parse each species' sepal lengths into NumPy arrays.
    setosa = np.fromstring(setosa_data, sep=" ", dtype=float)
    versicolor = np.fromstring(versicolor_data, sep=" ", dtype=float)
    virginica = np.fromstring(virginica_data, sep=" ", dtype=float)

    # Combine the measurements in the required species order.
    sepal_lengths = np.concatenate(
        (setosa, versicolor, virginica)
    )

    # Create the design matrix: column 0 is the bias, column 1 is sepal length.
    X = np.column_stack((
        np.ones(sepal_lengths.size, dtype=float),
        sepal_lengths
    ))

    # Integer labels: 0 = setosa, 1 = versicolor, 2 = virginica.
    y = np.concatenate((
        np.zeros(setosa.size, dtype=int),
        np.ones(versicolor.size, dtype=int),
        np.full(virginica.size, 2, dtype=int)
    ))

    return X, y

