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

# Step 17 - laplace_posterior
def laplace_posterior(X, y, prior_var):
    # Number of observations and features.
    N, D = X.shape

    # Convert the Gaussian-prior MAP objective to the averaged-NLL
    # objective used by fit_map_newton:
    # lambda = 1 / (2 * N * prior_var).
    lam = 1.0 / (2.0 * N * prior_var)

    # Compute the maximum a posteriori (MAP) estimate.
    w_map = fit_map_newton(X, y, lam)

    # Construct the posterior precision matrix, which is the Hessian
    # of the summed negative log-posterior at the MAP estimate.
    precision = (
        N * logreg_hessian(w_map, X)
        + np.eye(D) / prior_var
    )

    # The Laplace covariance is the inverse posterior precision.
    Sigma = np.linalg.inv(precision)

    return w_map, Sigma

# Step 18 - predictive_mc
def predictive_mc(X_test, w_map, Sigma, n_samples, rng):
    # Number of features.
    D = w_map.shape[0]

    # Compute the Cholesky factor of the posterior covariance matrix.
    L = np.linalg.cholesky(Sigma)

    # Draw all standard-normal samples in exactly one RNG call.
    Z = rng.standard_normal((n_samples, D))

    # Transform the samples to obtain weights from N(w_map, Sigma).
    Ws = w_map + Z @ L.T

    # Compute predictions for every test point under every sampled weight vector.
    # X_test has shape (M, D), and Ws.T has shape (D, n_samples).
    probabilities = sigmoid(X_test @ Ws.T)

    # Average over posterior samples to obtain the predictive probabilities.
    return np.mean(probabilities, axis=1)

# Step 19 - predictive_probit
def predictive_probit(X_test, w_map, Sigma):
    # Compute the mean logit for each test example.
    m = X_test @ w_map

    # Compute the posterior variance x^T Sigma x for every test example.
    # This vectorized calculation avoids Python loops.
    v = np.sum((X_test @ Sigma) * X_test, axis=1)

    # Apply the probit approximation correction for posterior uncertainty.
    kappa = 1.0 / np.sqrt(1.0 + np.pi * v / 8.0)

    # Return the approximate posterior predictive probabilities.
    return sigmoid(kappa * m)

# Step 20 - calibration_metrics
def calibration_metrics(p, y, n_bins=10):
    # Convert inputs to NumPy arrays and clip probabilities for numerical stability.
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    # Compute the mean binary cross-entropy (log loss).
    nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # Assign each prediction to an equal-width bin on [0, 1].
    # The final bin includes probability 1.0.
    bin_indices = np.minimum(
        np.floor(p * n_bins).astype(int),
        n_bins - 1
    )

    # Compute expected calibration error (ECE).
    ece = 0.0
    M = len(p)

    for b in range(n_bins):
        # Select predictions belonging to the current bin.
        mask = bin_indices == b

        # Empty bins contribute nothing to ECE.
        if np.any(mask):
            # Compare the observed positive frequency with the
            # average predicted probability in this bin.
            accuracy = np.mean(y[mask])
            confidence = np.mean(p[mask])

            # Weight the bin's calibration error by its fraction of samples.
            ece += (np.sum(mask) / M) * abs(accuracy - confidence)

    # Return both metrics as Python floats.
    return float(nll), float(ece)

# Step 21 - small_data_study
def small_data_study(
    n_tasks, n_train, dim, true_var, prior_var, rng,
    n_test=200, n_samples=1000
):
    # Accumulate log loss and ECE for each prediction method.
    scores = {
        "plugin": [],
        "probit": [],
        "mc": []
    }

    for _ in range(n_tasks):
        # 1. Draw the true weight vector.
        w_true = rng.normal(0.0, np.sqrt(true_var), size=dim)

        # 2. Draw all feature vectors (no bias column).
        X = rng.normal(size=(n_train + n_test, dim))

        # 3. Generate binary labels using the true logistic model.
        probabilities = sigmoid(X @ w_true)
        y = (rng.random(n_train + n_test) < probabilities).astype(int)

        # 4. Split the first n_train observations for training.
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]

        # 5. Fit the Laplace approximation to the posterior.
        w_map, Sigma = laplace_posterior(
            X_train, y_train, prior_var
        )

        # 6. Compute predictions in the required order.
        # Plug-in prediction uses only the MAP estimate.
        p_plugin = sigmoid(X_test @ w_map)

        # Probit prediction accounts approximately for posterior uncertainty.
        p_probit = predictive_probit(X_test, w_map, Sigma)

        # Monte Carlo prediction averages over sampled posterior weights.
        p_mc = predictive_mc(
            X_test, w_map, Sigma, n_samples, rng
        )

        # 7. Score each method and accumulate its metrics.
        scores["plugin"].append(
            calibration_metrics(p_plugin, y_test)
        )
        scores["probit"].append(
            calibration_metrics(p_probit, y_test)
        )
        scores["mc"].append(
            calibration_metrics(p_mc, y_test)
        )

    # Average (NLL, ECE) across tasks and return Python-float tuples.
    results = {}

    for method, task_scores in scores.items():
        mean_scores = np.mean(np.asarray(task_scores, dtype=float), axis=0)
        results[method] = (
            float(mean_scores[0]),
            float(mean_scores[1])
        )

    return results

# Step 22 - inject_label_noise
def inject_label_noise(X, y, w_ref, n_flip, mode):
    # Compute the signed margins relative to the reference weight vector.
    margins = X @ w_ref

    # Copy labels so the original y array is never modified.
    y_noisy = np.array(y, copy=True)

    if mode == "near":
        # Select the n_flip examples closest to the decision boundary,
        # regardless of their class. Stable sorting breaks ties by index.
        indices = np.argsort(np.abs(margins), kind="stable")[:n_flip]

    elif mode == "far":
        # Consider only class-0 examples.
        class0_indices = np.flatnonzero(y == 0)

        # Sort their margins in ascending order, so the most negative
        # margins (farthest into class 0) come first.
        sorted_positions = np.argsort(
            margins[class0_indices], kind="stable"
        )
        indices = class0_indices[sorted_positions[:n_flip]]

    else:
        # Reject unsupported modes.
        raise ValueError("mode must be 'near' or 'far'")

    # Flip the selected labels: 0 becomes 1 and 1 becomes 0.
    y_noisy[indices] = 1 - y_noisy[indices]

    return y_noisy

# Step 23 - mixture_nll
def mixture_nll(w, pi, X, y):
    # Compute the logistic model's predicted probabilities.
    mu = sigmoid(X @ w)

    # Compute the probability assigned by the logistic model to each
    # observed label: mu for label 1, and 1 - mu for label 0.
    q = np.where(y == 1, mu, 1.0 - mu)

    # Combine the label-noise process (a fair coin) with the logistic model.
    p = pi / 2.0 + (1.0 - pi) * q

    # Return the average negative log-likelihood as a Python float.
    return float(-np.mean(np.log(p)))

# Step 24 - clean_responsibility
def clean_responsibility(w, pi, X, y):
    # Compute the logistic model's predicted probability for class 1.
    mu = sigmoid(X @ w)

    # Compute the probability the model assigns to each observed label.
    # For y = 1, use mu; for y = 0, use 1 - mu.
    q = np.where(y == 1, mu, 1.0 - mu)

    # Compute the total mixture probability of each observed label.
    p = pi / 2.0 + (1.0 - pi) * q

    # Compute the posterior probability that each label came from
    # the logistic model rather than the noise process.
    r = (1.0 - pi) * q / p

    return r

# Step 25 - fit_robust_logreg
def fit_robust_logreg(X, y, lam, n_iter=100, pi0=0.1, n_newton=3):
    # Initialize the weights and label-noise probability.
    N, D = X.shape
    w = np.zeros(D, dtype=float)
    pi = float(pi0)

    # Construct the penalty matrix, excluding the bias at index 0.
    P = np.eye(D)
    P[0, 0] = 0.0

    history = []

    for _ in range(n_iter):
        # E-step: posterior probability that each observed label is clean.
        r = clean_responsibility(w, pi, X, y)

        # M-step for the noise probability.
        pi = float(1.0 - np.mean(r))

        # M-step for the weights: perform the requested Newton updates
        # on the responsibility-weighted penalized negative log-likelihood.
        for _ in range(n_newton):
            # Recompute probabilities at the current weights before each step.
            mu = sigmoid(X @ w)

            # Gradient of the weighted average NLL plus L2 penalty.
            g = (
                X.T @ (r * (mu - y)) / N
                + 2.0 * lam * (P @ w)
            )

            # Hessian of the weighted average NLL plus L2 penalty.
            weights = r * mu * (1.0 - mu)
            H = (
                (X.T * weights) @ X / N
                + 2.0 * lam * P
            )

            # Newton update.
            step = np.linalg.solve(H, g)
            w = w - step

        # Record the penalized mixture NLL after each EM iteration.
        penalty = lam * np.dot(w[1:], w[1:])
        history.append(mixture_nll(w, pi, X, y) + penalty)

    # Return final weights, noise probability, and history.
    return w, pi, np.asarray(history, dtype=float)

# Step 26 - log_t
def log_t(x, t):
    # Convert input to a floating-point NumPy array.
    x = np.asarray(x, dtype=float)

    # Use the standard logarithm when t is exactly 1.
    if t == 1.0:
        return np.log(x)

    # Compute the tempered logarithm elementwise.
    return (np.power(x, 1.0 - t) - 1.0) / (1.0 - t)


def exp_t(x, t):
    # Convert input to a floating-point NumPy array.
    x = np.asarray(x, dtype=float)

    # Use the standard exponential when t is exactly 1.
    if t == 1.0:
        return np.exp(x)

    # Compute the base of the tempered exponential.
    base = 1.0 + (1.0 - t) * x

    # The tempered exponential is zero when the base is non-positive.
    # Replacing invalid bases before exponentiation also avoids
    # zero raised to a negative power when t > 1.
    positive = base > 0.0
    safe_base = np.where(positive, base, 1.0)

    result = np.where(
        positive,
        np.power(safe_base, 1.0 / (1.0 - t)),
        0.0
    )

    return result

# Step 27 - tempered_softmax
def tempered_softmax(A, t, n_iter=100, tol=1e-12):
    # Ensure the input is a 2-D floating-point array.
    # A 1-D input is treated as a single row.
    A = np.atleast_2d(np.asarray(A, dtype=float))

    # For t = 1, compute the ordinary numerically stable softmax.
    if t == 1.0:
        shifted = A - np.max(A, axis=1, keepdims=True)
        exp_A = np.exp(shifted)
        return exp_A / np.sum(exp_A, axis=1, keepdims=True)

    # Shift each row by its maximum for numerical stability.
    mu = np.max(A, axis=1, keepdims=True)
    a0 = A - mu

    # Initialize the fixed-point iteration.
    a_prime = a0.copy()

    for _ in range(n_iter):
        # Sum the tempered exponentials for each row.
        z = np.sum(exp_t(a_prime, t), axis=1, keepdims=True)

        # Update the shifted logits using Algorithm 10.2.
        a_new = np.power(z, 1.0 - t) * a0

        # Keep the latest update and stop when the maximum change is small.
        converged = np.max(np.abs(a_new - a_prime)) < tol
        a_prime = a_new

        if converged:
            break

    # Recompute the normalization constant using the final iterate.
    z = np.sum(exp_t(a_prime, t), axis=1, keepdims=True)

    # Recover the row-wise normalization offset.
    lam = -log_t(1.0 / z, t) + mu

    # Return the normalized tempered probabilities.
    return exp_t(A - lam, t)

# Step 28 - bi_tempered_loss
def bi_tempered_loss(y, A, t1, t2):
    # Compute the tempered softmax probabilities using temperature t2.
    P = tempered_softmax(A, t2)

    # Select the probability assigned to the true class for each example.
    n = np.arange(len(y))
    true_class_probs = P[n, y]

    # Compute the tempered cross-entropy term.
    loss_log = -log_t(true_class_probs, t1)

    # Compute the correction term across all classes for each example.
    correction = (
        1.0 - np.sum(P ** (2.0 - t1), axis=1)
    ) / (2.0 - t1)

    # Average the bi-tempered loss over all examples.
    loss = loss_log - correction

    return float(np.mean(loss))

# Step 29 - bi_tempered_grad
def bi_tempered_grad(y, A, t1, t2):
    # Compute tempered softmax probabilities.
    P = tempered_softmax(A, t2)

    # Number of examples.
    N = len(y)

    # Compute the first term P_j^(1 - t1 + t2) for every class.
    v = P ** (1.0 - t1 + t2)

    # For each example, subtract P_c^(t2 - t1) at the true-class index.
    # This combines the label-dependent terms without separately
    # evaluating P_c^(-t1).
    rows = np.arange(N)
    true_probs = P[rows, y]
    v[rows, y] -= true_probs ** (t2 - t1)

    # Compute the normalized tempered-softmax Jacobian factors.
    P_t2 = P ** t2
    q = P_t2 / np.sum(P_t2, axis=1, keepdims=True)

    # Apply the Jacobian-vector product for each example.
    v_sum = np.sum(v, axis=1, keepdims=True)
    grad = v - q * v_sum

    # The loss is averaged over examples.
    return grad / N

