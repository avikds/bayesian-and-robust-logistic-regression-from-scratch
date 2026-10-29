# Bayesian and Robust Logistic Regression from Scratch

Logistic regression is the model everyone thinks they know, until the data is small, separable or mislabeled. Build it from scratch in NumPy following chapter 10 of Murphy's Probabilistic Machine Learning: derive and check the gradient and Hessian, fit Iris by SGD and by iteratively reweighted least squares, tame polynomial features with weight decay, and fit softmax regression with Bohning's bound. Then go Bayesian: a Laplace posterior on separable data, Monte Carlo and probit predictives, and a 100-task study of when averaging over weights beats the plug-in with only twenty labels. Finish with robust logistic regression: a label-noise mixture likelihood fitted by EM on contaminated Iris, and the bi-tempered loss against label noise far from and near the decision boundary.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** sigmoid
- [x] **2.** logreg_nll
- [x] **3.** logreg_grad
- [x] **4.** logreg_hessian
- [x] **5.** numerical_gradient
- [x] **6.** load_iris_sepal
- [x] **7.** sgd_logreg
- [x] **8.** irls_logreg
- [x] **9.** fit_map_newton
- [x] **10.** make_ring_data
- [x] **11.** polynomial_features
- [x] **12.** weight_decay_sweep
- [x] **13.** softmax_loglik_grad
- [x] **14.** bohning_bound
- [x] **15.** fit_softmax_mm
- [x] **16.** make_separable_data
- [x] **17.** laplace_posterior
- [x] **18.** predictive_mc
- [x] **19.** predictive_probit
- [x] **20.** calibration_metrics
- [x] **21.** small_data_study
- [x] **22.** inject_label_noise
- [x] **23.** mixture_nll
- [x] **24.** clean_responsibility
- [x] **25.** fit_robust_logreg
- [x] **26.** log_t
- [x] **27.** tempered_softmax
- [x] **28.** bi_tempered_loss
- [x] **29.** bi_tempered_grad
- [x] **30.** fit_bi_tempered

## Results

```
Derivative check at w=(-20, 3.5): max |grad - finite diff| = 4.1e-12, max |Hessian - finite diff| = 1.6e-10
IRLS: converged in 8 iterations, w = (-27.831, 5.140), NLL = 0.32106, decision boundary at 5.414 cm of sepal length
SGD lr=0.01: NLL after 1 / 10 / 200 epochs = 0.7196 / 0.6745 / 0.4927 (gap to the IRLS optimum after 20,000 updates: 0.1716)
SGD lr=0.1: NLL after 1 / 10 / 200 epochs = 1.0241 / 0.6296 / 0.3342 (gap to the IRLS optimum after 20,000 updates: 0.0131)
SGD lr=0.1 on the standardized feature: gap after 10 epochs = 0.0001 (the ill-conditioned raw loss surface is what slows SGD down)
Weight decay on degree-8 polynomial features (45 features, 50 training points, 5 seeds, 10% label noise):
  lambda=1e-06  train error 0.016  test error 0.223
  lambda=1e-05  train error 0.048  test error 0.207
  lambda=1e-04  train error 0.064  test error 0.189
  lambda=1e-03  train error 0.100  test error 0.173
  lambda=1e-02  train error 0.136  test error 0.182
  lambda=1e-01  train error 0.296  test error 0.339
  lambda=1e+00  train error 0.420  test error 0.417
  degree 1 at lambda=1e-04: train error 0.448  test error 0.488
  degree 2 at lambda=1e-04: train error 0.112  test error 0.184
Bohning MM, 2 classes: matches IRLS (w_0 = -w_irls: True), needs 48 iterations to get within 1e-6 nats of the optimum vs 8 for IRLS, but inverts its matrix once
Bohning MM, 3 Iris species from sepal length: log-likelihood -164.79 -> -91.03, never decreased: True, training accuracy 0.747
Separable data (30 points): ||w_map|| for lambda = 1e-2 / 1e-4 / 1e-6 = 2.17 / 6.28 / 12.07  (the MLE is at infinity)
Laplace posterior with sigma^2=100: w_map = (5.085, 2.600), posterior sd = (4.238, 3.563), correlation = 0.169
  x=(+1, +1): plug-in 0.9995  Monte Carlo 0.8906  probit 0.8788
  x=(-6, +6): plug-in 0.0000  Monte Carlo 0.3104  probit 0.3136
  x=(+4, -2): plug-in 1.0000  Monte Carlo 0.8100  probit 0.8014
  x=(-1, +2): plug-in 0.5288  Monte Carlo 0.5029  probit 0.5059
Plug-in vs Bayesian predictive, 100 tasks, 20 training labels, 5 features, true weights ~ N(0, 4 I):
  prior variance   100: plugin NLL 0.6814 ECE 0.1325  probit NLL 0.4287 ECE 0.0895  mc NLL 0.4274 ECE 0.0869
  prior variance     4: plugin NLL 0.4148 ECE 0.0760  probit NLL 0.4152 ECE 0.0846  mc NLL 0.4133 ECE 0.0825
Contaminated Iris (6 versicolor labels at 4.2-4.5 cm, Fig 10.10): boundary clean 5.414 cm, plain fit 5.309 cm, mixture fit 5.428 cm (pi = 0.211, max P(clean) of the six = 0.090)
Label noise far from the boundary (20 of 200 labels flipped, clean-test accuracy, mean of 3 seeds):
  logistic (t1=1, t2=1)    0.8998
  bounded loss (t1=0.2)    1.0000
  heavy tails (t2=4)       0.9997
  bi-tempered (0.2, 4)     0.9993
  mixture likelihood       1.0000
Label noise near the boundary (20 of 200 labels flipped, clean-test accuracy, mean of 3 seeds):
  logistic (t1=1, t2=1)    0.9967
  bounded loss (t1=0.2)    0.9955
  heavy tails (t2=4)       0.9982
  bi-tempered (0.2, 4)     0.9992
  mixture likelihood       0.9967
```
