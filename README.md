[![PyPI version](https://badge.fury.io/py/torch_linear_regression.svg)](https://badge.fury.io/py/torch_linear_regression)
[![Downloads](https://pepy.tech/badge/torch_linear_regression)](https://pepy.tech/project/torch_linear_regression)
[![repo size](https://img.shields.io/github/repo-size/RichieHakim/torch_linear_regression)](https://github.com/RichieHakim/torch_linear_regression/)

#  torch_linear_regression 
A very simple library containing closed-form linear regression models using PyTorch. Accepts both NumPy arrays and PyTorch tensors, follows the scikit-learn estimator interface (`fit`/`predict`/`score`), and can run on GPU.

### Models
- **OLS** -- Ordinary Least Squares: `(X'X)^-1 X'Y`
- **Ridge** -- Ridge regression: `(X'X + lambda*I)^-1 X'Y`. The penalty is a fixed scalar, or, with `prefit_includes_alpha=False`, can change on every fit and differ per column of Y
- **ReducedRankRidgeRegression** -- Ridge regression followed by SVD truncation of the weight matrix
- **RidgeMML** -- Ridge regression with automatic per-column lambda estimation ([Karabatsos 2017](http://www.tandfonline.com/doi/pdf/10.1080/03610918.2017.1321119)).

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
| Intercept | Optional | Always (derived from column means) |
| Dependencies | numpy, torch | numpy, torch |
