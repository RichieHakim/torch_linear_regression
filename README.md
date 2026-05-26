[![PyPI version](https://badge.fury.io/py/torch_linear_regression.svg)](https://badge.fury.io/py/torch_linear_regression)
[![Downloads](https://pepy.tech/badge/torch_linear_regression)](https://pepy.tech/project/torch_linear_regression)
[![repo size](https://img.shields.io/github/repo-size/RichieHakim/torch_linear_regression)](https://github.com/RichieHakim/torch_linear_regression/)

#  torch_linear_regression 
A very simple library containing closed-form linear regression models using PyTorch. Accepts both NumPy arrays and PyTorch tensors, follows the scikit-learn estimator interface (`fit`/`predict`/`score`), and can run on GPU.

### Models
- **OLS** -- Ordinary Least Squares: `(X'X)^-1 X'Y`
- **Ridge** -- Ridge regression with a fixed scalar penalty: `(X'X + lambda*I)^-1 X'Y`
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
to multiple targets.

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
| Lambda | Single scalar, user-specified | Per-column of Y, estimated via MML |
| X standardization | None | Intrinsic z-scoring (ddof=1) |
| Intercept | Optional | Always (derived from column means) |
| Dependencies | numpy, torch | numpy, torch |
