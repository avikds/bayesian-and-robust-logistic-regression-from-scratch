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
SETOSA_SEPAL_LENGTH = (
    "5.1 4.9 4.7 4.6 5.0 5.4 4.6 5.0 4.4 4.9 5.4 4.8 4.8 4.3 5.8 5.7 5.4 5.1 5.7 5.1 5.4 5.1 4.6 5.1 4.8 "
    "5.0 5.0 5.2 5.2 4.7 4.8 5.4 5.2 5.5 4.9 5.0 5.5 4.9 4.4 5.1 5.0 4.5 4.4 5.0 5.1 4.8 5.1 4.6 5.3 5.0"
)
VERSICOLOR_SEPAL_LENGTH = (
    "7.0 6.4 6.9 5.5 6.5 5.7 6.3 4.9 6.6 5.2 5.0 5.9 6.0 6.1 5.6 6.7 5.6 5.8 6.2 5.6 5.9 6.1 6.3 6.1 6.4 "
    "6.6 6.8 6.7 6.0 5.7 5.5 5.5 5.8 6.0 5.4 6.0 6.7 6.3 5.6 5.5 5.5 6.1 5.8 5.0 5.6 5.7 5.7 6.2 5.1 5.7"
)
VIRGINICA_SEPAL_LENGTH = (
    "6.3 5.8 7.1 6.3 6.5 7.6 4.9 7.3 6.7 7.2 6.5 6.4 6.8 5.7 5.8 6.4 6.5 7.7 7.7 6.0 6.9 5.6 7.7 6.3 6.7 "
    "7.2 6.2 6.1 6.4 7.2 7.4 7.9 6.4 6.3 6.1 7.7 6.3 6.4 6.0 6.9 6.7 6.9 5.8 6.8 6.7 6.7 6.3 6.5 6.2 5.9"
)


def load_iris_sepal():
    # Parse the sepal lengths for each class.
    setosa = np.fromstring(SETOSA_SEPAL_LENGTH, sep=" ")
    versicolor = np.fromstring(VERSICOLOR_SEPAL_LENGTH, sep=" ")
    virginica = np.fromstring(VIRGINICA_SEPAL_LENGTH, sep=" ")

    # Concatenate in the required order: setosa, versicolor, virginica.
    sepal_lengths = np.concatenate((setosa, versicolor, virginica))

    # Create integer labels: 0 for setosa, 1 for versicolor, 2 for virginica.
    y = np.repeat(np.array([0, 1, 2], dtype=int), 50)

    # Construct the design matrix with a bias column and sepal length.
    X = np.column_stack((np.ones(150), sepal_lengths))

    return X, y

# Step 7 - sgd_logreg
def sgd_logreg(X, y, lr, n_epochs, rng):
    # Get the number of training examples and features.
    N, D = X.shape

    # Initialize the weights to zero.
    w = np.zeros(D, dtype=float)

    # Store the average negative log-likelihood after each epoch.
    history = []

    for epoch in range(n_epochs):
        # Generate the visiting order with exactly one permutation per epoch.
        order = rng.permutation(N)

        # Perform single-example stochastic gradient descent.
        for n in order:
            # Compute the predicted probability for the current example.
            mu_n = sigmoid(X[n] @ w)

            # Update the weights using the gradient of this example's loss.
            w -= lr * (mu_n - y[n]) * X[n]

        # Record the average NLL after completing this epoch.
        history.append(logreg_nll(w, X, y))

    # Return the final weights and the loss history as a NumPy array.
    return w, np.asarray(history, dtype=float)

# Step 8 - irls_logreg
def irls_logreg(X, y, max_iter=50, tol=1e-10):
    # Initialize the weight vector to zero.
    w = np.zeros(X.shape[1], dtype=float)

    # Record the initial negative log-likelihood before any updates.
    history = [logreg_nll(w, X, y)]

    for _ in range(max_iter):
        # Compute logits, probabilities, and their associated weights.
        a = X @ w
        mu = sigmoid(a)
        s = mu * (1.0 - mu)

        # Avoid division by zero in extreme numerical cases.
        s = np.maximum(s, np.finfo(float).tiny)

        # Compute the working response for IRLS.
        z = a + (y - mu) / s

        # Construct the weighted least-squares system without forming
        # the N x N diagonal matrix S.
        XtS = X.T * s
        lhs = XtS @ X
        rhs = XtS @ z

        # Solve the weighted least-squares system for the new weights.
        w_new = np.linalg.solve(lhs, rhs)

        # Record the NLL after the update.
        history.append(logreg_nll(w_new, X, y))

        # Check convergence before replacing the current weights.
        converged = np.max(np.abs(w_new - w)) < tol
        w = w_new

        if converged:
            break

    # Return the final weights and the NLL history as a NumPy array.
    return w, np.asarray(history, dtype=float)

# Step 9 - fit_map_newton
def fit_map_newton(X, y, lam, max_iter=100, tol=1e-10):
    # Initialize the weights to zero.
    D = X.shape[1]
    w = np.zeros(D, dtype=float)
    I = np.eye(D)

    # Define the penalized negative log-likelihood objective.
    def objective(weights):
        return logreg_nll(weights, X, y) + lam * (weights @ weights)

    for _ in range(max_iter):
        # Compute the gradient and Hessian of the penalized objective.
        g = logreg_grad(w, X, y) + 2.0 * lam * w
        H = logreg_hessian(w, X) + 2.0 * lam * I

        # Solve the Newton system H @ step = g.
        step = np.linalg.solve(H, g)

        # Evaluate the current objective before backtracking.
        current_obj = objective(w)

        # Backtracking line search: halve eta while the objective increases.
        eta = 1.0
        while eta > 1e-10 and objective(w - eta * step) > current_obj:
            eta *= 0.5

        # Apply the accepted Newton update.
        update = eta * step
        w = w - update

        # Stop when the maximum absolute weight change is below tolerance.
        if np.max(np.abs(update)) < tol:
            break

    return w

# Step 10 - make_ring_data
def make_ring_data(n, flip_prob, rng):
    # Draw all feature vectors first, uniformly from [-1, 1]^2.
    X = rng.uniform(-1.0, 1.0, size=(n, 2))

    # Assign clean labels: 1 inside the circle and 0 outside.
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 0.5).astype(int)

    # Draw once for all examples, then flip labels where the draw is below flip_prob.
    flip_mask = rng.random(n) < flip_prob
    y[flip_mask] = 1 - y[flip_mask]

    return X, y

# Step 11 - polynomial_features
def polynomial_features(X, degree):
    # Extract the two input features.
    x1 = X[:, 0]
    x2 = X[:, 1]

    # Store the generated monomial columns in the required order.
    features = []

    # Enumerate monomials by increasing total degree.
    for d in range(degree + 1):
        # Within each degree, use decreasing powers of x1.
        for i in range(d, -1, -1):
            # Append x1^i * x2^(d-i).
            features.append((x1 ** i) * (x2 ** (d - i)))

    # Stack the monomial columns into an (N, M) array.
    return np.column_stack(features)

# Step 12 - weight_decay_sweep
def weight_decay_sweep(X_train, y_train, X_test, y_test, lams):
    # Store the training and test misclassification rates for each penalty.
    train_err = []
    test_err = []

    for lam in lams:
        # Fit logistic regression using the specified L2 penalty.
        w = fit_map_newton(X_train, y_train, lam)

        # Predict class 1 when the linear score is strictly positive.
        train_pred = (X_train @ w > 0).astype(int)
        test_pred = (X_test @ w > 0).astype(int)

        # Compute the misclassification rates.
        train_err.append(np.mean(train_pred != y_train))
        test_err.append(np.mean(test_pred != y_test))

    # Return NumPy arrays in the same order as the supplied penalties.
    return np.asarray(train_err, dtype=float), np.asarray(test_err, dtype=float)

# Step 13 - softmax_loglik_grad
def softmax_loglik_grad(W, X, y, C):
    # Compute logits for the C - 1 free classes.
    # W has shape (C - 1, D), so logits_free has shape (N, C - 1).
    logits_free = X @ W.T

    # The last class is the reference class, with its logits fixed at zero.
    reference_logits = np.zeros((X.shape[0], 1), dtype=float)
    logits = np.concatenate((logits_free, reference_logits), axis=1)

    # Compute a numerically stable log-sum-exp by shifting each row
    # by its maximum logit.
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - max_logits
    exp_logits = np.exp(shifted_logits)

    log_sum_exp = (
        max_logits[:, 0]
        + np.log(np.sum(exp_logits, axis=1))
    )

    # Compute the summed multinomial log-likelihood.
    # The selected class logit is gathered for each observation.
    n = np.arange(X.shape[0])
    ll = np.sum(logits[n, y] - log_sum_exp)

    # Compute the softmax probabilities for all C classes.
    mu = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Construct one-hot labels for the C - 1 free classes.
    Y_free = (y[:, None] == np.arange(C - 1)[None, :]).astype(float)

    # Compute the gradient of the log-likelihood with respect to
    # the free-class weights. Its shape is (C - 1, D).
    G = (Y_free - mu[:, :C - 1]).T @ X

    return float(ll), G

# Step 14 - bohning_bound
def bohning_bound(X, C):
    # Number of free classes (the last class is the reference class).
    K = C - 1

    # Construct I - 11^T / C for the free classes.
    class_matrix = np.eye(K) - np.ones((K, K)) / C

    # Compute X^T X.
    feature_matrix = X.T @ X

    # Construct Bohning's constant lower bound using the Kronecker product.
    # Class-major ordering matches W.reshape(-1) for W of shape (C - 1, D).
    B = -0.5 * np.kron(class_matrix, feature_matrix)

    return B

# Step 15 - fit_softmax_mm
def fit_softmax_mm(X, y, C, n_iter):
    # Initialize the free-class weights to zero.
    D = X.shape[1]
    W = np.zeros((C - 1, D), dtype=float)

    # Compute the inverse of Böhning's bound once.
    B_inv = np.linalg.inv(bohning_bound(X, C))

    # Record the initial log-likelihood before the first update.
    ll, G = softmax_loglik_grad(W, X, y, C)
    history = [ll]

    for _ in range(n_iter):
        # Flatten weights and gradient in class-major order.
        w_flat = W.reshape(-1)
        g_flat = G.reshape(-1)

        # Apply the Böhning-bound minorize-maximize update.
        w_flat = w_flat - B_inv @ g_flat

        # Restore the weight matrix shape (C - 1, D).
        W = w_flat.reshape(C - 1, D)

        # Evaluate the updated log-likelihood and gradient.
        ll, G = softmax_loglik_grad(W, X, y, C)

        # Record the log-likelihood after this update.
        history.append(ll)

    return W, np.asarray(history, dtype=float)

# Step 16 - make_separable_data
def make_separable_data(n, margin, rng):
    # Fixed unit direction for the separating line.
    u = np.array([2.0, 1.0]) / np.sqrt(5.0)

    # Draw all feature vectors in exactly one random-number generator call.
    X = rng.normal(0.0, 2.0, size=(n, 2))

    # Assign class 1 when the projection onto u is positive; otherwise class 0.
    y = (X @ u > 0.0).astype(int)

    # Push each point away from the separating line by the specified margin.
    # Class 1 moves along +u, while class 0 moves along -u.
    X = X + margin * np.outer(2 * y - 1, u)

    return X, y

