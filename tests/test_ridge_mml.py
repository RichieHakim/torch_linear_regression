"""
Tests for RidgeMML (ridge regression with per-column MML lambda estimation).

NOTE: RidgeMML is being implemented in parallel. These tests assume:
  - `from torch_linear_regression import RidgeMML` works
    (i.e., __init__.py has been updated to export RidgeMML).
  - The class API matches the spec in the briefing.
"""

import pytest
from sklearn.datasets import make_regression
import numpy as np
import torch

from torch_linear_regression import RidgeMML


# ---------------------------------------------------------------------------
# 1. Synthetic data, basic fit
# ---------------------------------------------------------------------------
def test_basic_fit_r2():
    """Fit on synthetic data, verify R^2 > 0.8."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_targets=5,
        noise=0.1,
        random_state=42,
    )
    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    r2 = model.score(X, y)
    assert r2 > 0.8, f"R^2 = {r2}, expected > 0.8"


# ---------------------------------------------------------------------------
# 2. Shape checks
# ---------------------------------------------------------------------------
def test_shapes():
    """Verify coef_, intercept_, lambdas_, convergence_failures_ shapes."""
    n, p_x, p_y = 200, 20, 5
    X, y = make_regression(
        n_samples=n,
        n_features=p_x,
        n_targets=p_y,
        random_state=0,
    )
    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    assert model.coef_.shape == (p_x, p_y)
    assert model.intercept_.shape == (p_y,)
    assert model.lambdas_.shape == (p_y,)
    assert model.convergence_failures_.shape == (p_y,)


# ---------------------------------------------------------------------------
# 3. fit_intercept=False raises NotImplementedError
# ---------------------------------------------------------------------------
def test_fit_intercept_false_raises():
    """fit_intercept=False should raise NotImplementedError."""
    X, y = make_regression(n_samples=50, n_features=5, random_state=0)
    with pytest.raises(NotImplementedError):
        model = RidgeMML(fit_intercept=False)
        model.fit(X, y)


# ---------------------------------------------------------------------------
# 4. Pre-supplied lambdas
# ---------------------------------------------------------------------------
def test_presupplied_lambdas():
    """When lambdas are provided, model.lambdas_ should equal the input."""
    n, p_x, p_y = 200, 10, 3
    X, y = make_regression(
        n_samples=n,
        n_features=p_x,
        n_targets=p_y,
        random_state=7,
    )
    lambdas_input = np.array([1.0, 10.0, 100.0])
    model = RidgeMML(fit_intercept=True, lambdas=lambdas_input)
    model.fit(X, y)

    np.testing.assert_allclose(model.lambdas_, lambdas_input)


# ---------------------------------------------------------------------------
# 5. 1D y
# ---------------------------------------------------------------------------
def test_1d_y():
    """Fit with 1D y, check it works and shapes are correct."""
    n, p_x = 200, 10
    X, y = make_regression(
        n_samples=n,
        n_features=p_x,
        random_state=1,
    )
    assert y.ndim == 1  ## make_regression returns 1D when n_targets=1

    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    ## coef_ should be (p_x, 1) or (p_x,) — spec says (p_x, p_y) with p_y=1
    assert model.coef_.shape == (p_x, 1)
    assert model.intercept_.shape == (1,)
    assert model.lambdas_.shape == (1,)

    y_pred = model.predict(X)
    assert y_pred.shape[0] == n


# ---------------------------------------------------------------------------
# 6. np input -> np output
# ---------------------------------------------------------------------------
def test_np_input_np_output():
    """When X is np.ndarray, coef_ should be np.ndarray."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_targets=3,
        random_state=2,
    )
    assert isinstance(X, np.ndarray)

    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    assert isinstance(model.coef_, np.ndarray)
    assert isinstance(model.intercept_, np.ndarray)


# ---------------------------------------------------------------------------
# 7. torch input -> torch output
# ---------------------------------------------------------------------------
def test_torch_input_torch_output():
    """When X is torch.Tensor, coef_ should be torch.Tensor."""
    X_np, y_np = make_regression(
        n_samples=100,
        n_features=10,
        n_targets=3,
        random_state=3,
    )
    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)

    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    assert isinstance(model.coef_, torch.Tensor)
    assert isinstance(model.intercept_, torch.Tensor)


# ---------------------------------------------------------------------------
# 8. Intercept correctness
# ---------------------------------------------------------------------------
def test_intercept_correctness():
    """Generate data with known intercept, verify model.intercept_ is close."""
    rng = np.random.RandomState(99)
    n, p_x, p_y = 500, 10, 3
    X = rng.randn(n, p_x)
    true_coef = rng.randn(p_x, p_y)
    true_intercept = np.array([5.0, -3.0, 10.0])
    noise = rng.randn(n, p_y) * 0.1

    y = X @ true_coef + true_intercept[None, :] + noise

    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    ## Ridge shrinkage biases coefficients, so use generous atol
    np.testing.assert_allclose(model.intercept_, true_intercept, atol=0.5)


# ---------------------------------------------------------------------------
# 9. predict matches manual computation
# ---------------------------------------------------------------------------
def test_predict_matches_manual():
    """X @ coef_ + intercept_ should match predict(X)."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_targets=3,
        random_state=4,
    )
    model = RidgeMML(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_manual = X @ model.coef_ + model.intercept_

    np.testing.assert_allclose(y_pred, y_manual, rtol=1e-10)


# ---------------------------------------------------------------------------
# 10. batch_size_solve gives same result as unbatched
# ---------------------------------------------------------------------------
def test_batch_size_solve():
    """batch_size_solve=2 should give same result as None (all at once)."""
    X, y = make_regression(
        n_samples=200,
        n_features=15,
        n_targets=6,
        random_state=5,
    )
    model_full = RidgeMML(fit_intercept=True, batch_size_solve=None)
    model_full.fit(X, y)

    model_batched = RidgeMML(fit_intercept=True, batch_size_solve=2)
    model_batched.fit(X, y)

    np.testing.assert_allclose(model_full.coef_, model_batched.coef_, rtol=1e-5)
    np.testing.assert_allclose(model_full.intercept_, model_batched.intercept_, rtol=1e-5)
    np.testing.assert_allclose(model_full.lambdas_, model_batched.lambdas_, rtol=1e-5)


# ---------------------------------------------------------------------------
# 11. Numerical correctness against independent numpy reference
# ---------------------------------------------------------------------------
def test_coef_intercept_against_numpy_reference():
    """
    With pre-supplied lambdas, verify coef_ and intercept_ match an
    independent numpy closed-form solution. This catches algebra errors
    in z-scoring, un-z-scoring, and intercept derivation.
    """
    rng = np.random.RandomState(77)
    n, p_x, p_y = 300, 15, 4
    X = rng.randn(n, p_x) * rng.uniform(0.5, 5.0, size=p_x)
    X += rng.randn(p_x) * 10  ## non-zero column means
    true_coef = rng.randn(p_x, p_y)
    true_intercept = rng.randn(p_y) * 5
    Y = X @ true_coef + true_intercept[None, :] + rng.randn(n, p_y) * 0.5

    ## Fixed lambdas for deterministic reference
    lambdas = np.array([1.0, 10.0, 100.0, 0.1])

    ## Independent numpy reference
    X_std = np.std(X, axis=0, ddof=1)
    X_std[X_std == 0] = 1.0
    X_z = X / X_std[None, :]
    X_z_centered = X_z - X_z.mean(axis=0)[None, :]
    Y_centered = Y - Y.mean(axis=0)[None, :]

    coef_expected = np.empty((p_x, p_y))
    for i in range(p_y):
        A = X_z_centered.T @ X_z_centered + lambdas[i] * np.eye(p_x)
        coef_expected[:, i] = np.linalg.solve(A, X_z_centered.T @ Y_centered[:, i])
    coef_expected /= X_std[:, None]
    intercept_expected = Y.mean(axis=0) - X.mean(axis=0) @ coef_expected

    ## RidgeMML with same lambdas
    model = RidgeMML(fit_intercept=True, lambdas=lambdas)
    model.fit(X, Y)

    np.testing.assert_allclose(model.coef_, coef_expected, rtol=1e-10)
    np.testing.assert_allclose(model.intercept_, intercept_expected, rtol=1e-10)
