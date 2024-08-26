[![PyPI version](https://badge.fury.io/py/torch_linear_regression.svg)](https://badge.fury.io/py/torch_linear_regression)
[![Downloads](https://pepy.tech/badge/torch_linear_regression)](https://pepy.tech/project/torch_linear_regression)
[![repo size](https://img.shields.io/github/repo-size/RichieHakim/torch_linear_regression)](https://github.com/RichieHakim/torch_linear_regression/)

#  torch_linear_regression 
A very simple library containing closed-form linear regression models using PyTorch.
Includes:
- Ordinary Least Squares (OLS) Linear Regression: `(X'X)^-1 X'Y`
- Ridge Regression: `(X'X + λI)^-1 X'Y`
- Reduced Rank Regression (RRR) with Ridge penalty: Ridge regression followed by SVD on the weights matrix

The closed-form approach results in fast and accurate results under most
conditions. However, when ``n_features`` is large and/or very underdetermined
(``n_samples`` << ``n_features``), the closed-form solution will start to
diverge from other solutions. Also, if the input `X` matrix is singular (has
redundant columns), an error will be thrown. If you encounter these issues,
either consider using SVD / PCA to reduce the redundancy in your input matrix,
or just let me know and I will implement the least squares solution for these
edge cases.

Each model also includes a `model.prefit()` method that can be used to precompute
the inverse matrix and the ridge penalty matrix. This can be useful when the model
is used multiple times with the same `X` input data.

Because the models are based on PyTorch, they are significantly faster than sklearn's
models, and can be further accelerated by using a GPU. Also the models can be used
in conjunction with PyTorch's autograd for gradient-based optimization.

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
See the notebook for more examples: [demo notebook](https://github.com/RichieHakim/torch_linear_regression/blob/master/demo_notebook.ipynb)
```
import torch_linear_regression as tlr

import torch
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt


## Generate data for regression
X, Y = sklearn.datasets.make_regression(
    n_samples=100,
    n_features=2,
    n_informative=10,
    bias=2,
    noise=50,
    random_state=42,
)

## Create model
model_ols = tlr.OLS()
## Fit model
model_ols.fit(X=X, y=Y)
## Predict
Y_pred = model_ols.predict(X)
## Score model
score = model_ols.score(X=X, y=Y)
print(f"R^2: {score}")
```