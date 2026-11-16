[![PyPI version](https://badge.fury.io/py/torch_linear_regression.svg)](https://badge.fury.io/py/torch_linear_regression)
[![Downloads](https://pepy.tech/badge/torch_linear_regression)](https://pepy.tech/project/torch_linear_regression)
[![repo size](https://img.shields.io/github/repo-size/RichieHakim/torch_linear_regression)](https://github.com/RichieHakim/torch_linear_regression/)

#  torch_linear_regression 
A very simple library containing closed-form linear regression models using PyTorch. Accepts both NumPy arrays and PyTorch tensors, follows the scikit-learn estimator interface (`fit`/`predict`/`score`), and can run on GPU.

### Models
- **OLS** -- Ordinary Least Squares: `(X'X)^-1 X'Y`
- **Ridge** -- Ridge regression: `(X'X + lambda*I)^-1 X'Y`. The penalty is a fixed scalar, or, with `prefit_includes_alpha=False`, can change on every fit and differ per column of Y. The intercept is not penalized, as in scikit-learn's `Ridge`
- **ReducedRankRidgeRegression** -- Ridge regression followed by SVD truncation of the weight matrix. The intercept is not penalized
- **RidgeMML** -- Ridge regression with automatic per-column lambda estimation ([Karabatsos 2017](http://www.tandfonline.com/doi/pdf/10.1080/03610918.2017.1321119)).
- **lars_elastic_net_path** -- Exact elastic-net path for a fixed L2 penalty, by least angle regression with the lasso modification (LARS-EN; Zou & Hastie 2005). Records the order in which features enter and drop out.

The closed-form approach results in fast and accurate results under most
conditions. However, when `n_features` is large and/or very underdetermined
(`n_samples` << `n_features`), the closed-form solution will start to
diverge from other solutions. Also, if the input `X` matrix is singular (has
redundant columns), an error will be thrown. If you encounter these issues,
consider using SVD / PCA to reduce the redundancy in your input matrix.

OLS, Ridge, and ReducedRankRidgeRegression include a `model.prefit()` method
that precomputes the inverse matrix. This is useful when fitting the same `X`
to multiple targets. Ridge's default prefit fixes the penalty;
`prefit_includes_alpha=False` instead precomputes an SVD of `X` and takes the
penalty at each fit (see below).

## Installation
Install stable version:
```
pip install torch_linear_regression
```

Install development version:
```
pip install git+https://github.com/RichieHakim/torch_linear_regression.git
```

## Usage 

### OLS / Ridge
See the [demo notebook](https://github.com/RichieHakim/torch_linear_regression/blob/master/demo_notebook.ipynb)
for more examples.
```python
import torch_linear_regression as tlr
import numpy as np
from sklearn.datasets import make_regression

X, Y = make_regression(n_samples=100, n_features=10, noise=5, random_state=42)

model = tlr.OLS()
model.fit(X=X, y=Y)
Y_pred = model.predict(X)
print(f"R^2: {model.score(X=X, y=Y):.4f}")
```

### Ridge penalty search
With `prefit_includes_alpha=False`, Ridge precomputes the thin SVD of `X`,
which does not depend on the penalty. Each `fit` then takes `alpha` as a
scalar or as one value per column of Y, and costs only small matrix products.
The SVD is also far smaller than the `(n_features, n_features)` inverse when
`n_features > n_samples`. Both prefit modes give the same solution for the
same `alpha`.

In both modes, the intercept is not penalized. With `fit_intercept=True`
(the default), the columns of `X` and `Y` are centered before the solve, and
the intercept is `mean(Y) - mean(X) @ coef`. The results match
`sklearn.linear_model.Ridge`. Version 0.2.0 and earlier fit the intercept as a
penalized column of ones, which shrank it toward zero when `alpha` was large
compared with `n_samples`.

```python
X = np.random.randn(1000, 5000)
Y = X[:, :50] @ np.random.randn(50, 100) + np.random.randn(1000, 100)
X_train, X_test, Y_train, Y_test = X[:900], X[900:], Y[:900], Y[900:]

model = tlr.Ridge(prefit_X=X_train, prefit_includes_alpha=False)
for alpha in [1e1, 1e2, 1e3, 1e4]:
    model.fit(X_train, Y_train, alpha=alpha)
    print(f"alpha={alpha:.0e}  held-out R^2: {model.score(X_test, Y_test):.3f}")

model.fit(X_train, Y_train, alpha=np.logspace(1, 4, 100))  ## one penalty per column of Y
```

### RidgeMML
Ridge regression where each column of Y gets its own regularization
parameter, estimated automatically via marginal maximum likelihood.

X columns are z-scored internally (required by the algorithm for lambda
comparability across features). Coefficients are un-z-scored before
storage, so `predict(X_raw)` works on raw data.

```python
import torch_linear_regression as tlr
import numpy as np

## Multi-target regression
X = np.random.randn(500, 20)
beta_true = np.random.randn(20, 5)
Y = X @ beta_true + 0.5 * np.random.randn(500, 5)

model = tlr.RidgeMML()
model.fit(X, Y)

print(f"R^2: {model.score(X, Y):.4f}")
print(f"Lambdas: {model.lambdas_}")          ## per-column regularization
print(f"Converged: {~model.convergence_failures_.any()}")

## Pre-supplied lambdas (skip MML optimization)
model_fixed = tlr.RidgeMML(lambdas=np.ones(5) * 10.0)
model_fixed.fit(X, Y)

## GPU acceleration
model_gpu = tlr.RidgeMML(device="cuda")
model_gpu.fit(X, Y)

## Control memory for large p_y
model_batched = tlr.RidgeMML(batch_size_solve=50)
model_batched.fit(X, Y)
```

Key differences from `Ridge`:

| | `Ridge` | `RidgeMML` |
|---|---|---|
| Lambda | User-specified: one scalar, or one per column of Y with `prefit_includes_alpha=False` | Per-column of Y, estimated via MML |
| X standardization | None | Intrinsic z-scoring (ddof=1) |
| Intercept | Optional, not penalized | Always (derived from column means) |
| Dependencies | numpy, torch | numpy, torch |

### LARS elastic-net path
`lars_elastic_net_path` minimizes
`(1/(2n))||y - Xb||^2 + lambda1*||b||_1 + (lambda2/2)*||b||^2` for a fixed
`lambda2` and every `lambda1`, from the value where the first feature enters
down to `lambda1_min`. This is the convention of sklearn's `ElasticNet`
(`alpha*l1_ratio = lambda1`, `alpha*(1 - l1_ratio) = lambda2`). It returns one
event per step (the feature that entered or dropped, and its sign), `lambda1`
at each knot, and the coefficients at each knot as active indices plus values.
Events that happen at the same `lambda1` (for example, exact copies of a
column) share a value of `events_group`; their order within the group is
arbitrary. Each step makes one product with `X` and never forms the
`(n_features, n_features)` Gram, so it suits very wide `X` when only the first
few dozen steps are needed. It fits one target, with no centering or
intercept: center `X` and `y` first. Use float64 when the order of features
matters; float32 can swap features that enter within about 1e-4 of `lambda1`
of each other. Tested against sklearn's `lars_path`, `lars_path_gram` and
`ElasticNet`, and against R's `elasticnet::enet`, in float64.

```python
X = np.random.randn(200, 5000)
y = X[:, :5] @ np.random.randn(5) + np.random.randn(200)
X, y = X - X.mean(axis=0), y - y.mean()

path = tlr.lars_elastic_net_path(X=X, y=y, lambda2=0.1, max_steps=30)
print(path['events_index'], path['events_type'])  ## feature order; +1 entered, -1 dropped
print(path['events_sign'], path['events_group'])  ## coefficient sign; equal groups = same lambda1
print(path['lambda1s'])  ## lambda1 at each knot
lambda1 = 0.5 * (path['lambda1s'][5] + path['lambda1s'][6]).item()  ## between knots 5 and 6
coef = tlr.interpolate_lars_path(path, lambda1=lambda1)  ## dense, shape (n_features,)
```

If the active-set Gram becomes singular (duplicate or collinear columns with
`lambda2 = 0`), the function raises `tlr.SingularGramError`, a `ValueError`
whose `path` attribute holds the path computed up to that point.

Version 0.4.0 added `lars_elastic_net_path` and `interpolate_lars_path`.
