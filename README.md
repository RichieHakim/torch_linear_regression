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
conditions. However, when ``n_features`` is large and/or underdetermined
(``n_samples`` <= ``n_features``), the solution will start to diverge from
gradient-based / sklearn solutions.

Each model also includes a `model.prefit()` method that can be used to precompute
the inverse matrix and the ridge penalty matrix. This can be useful when the model
is used multiple times with the same `X` input data.

Because the models are based on PyTorch, they are significantly faster than sklearn's
models, and can be further accelerated by using GPU. Also the models can be used
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
```
import torch_linear_regression as tlr

# Generate some data
X = torch.randn(100, 10)
Y = torch.randn(100, 2)

# Fit OLS model
ols = tlr.OLS()
ols.fit(X, Y)
Y_pred = ols.predict(X)


```