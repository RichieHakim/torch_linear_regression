"""
Tests for Ridge and ReducedRankRidgeRegression against sklearn's Ridge, whose
intercept is not penalized, and for Ridge with ``prefit_includes_alpha=False``:
a prefit of the thin SVD of X that does not depend on alpha, so that alpha can
be passed to each fit.
"""

import warnings

import pytest
import numpy as np
import sklearn.linear_model
import torch

from torch_linear_regression import Ridge, ReducedRankRidgeRegression


SHAPES = [(100, 20), (40, 150)]  ## (n_samples, n_features): p < n and p > n
ALPHAS = [1e-2, 1.0, 1e2, 1e4]
N_TARGETS = 4
TOLERANCES = dict(rtol=1e-6, atol=1e-9)


def make_data(n_samples, n_features, n_targets=N_TARGETS, seed=0):
    """Random X and y with nonzero means, so that the intercept matters."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)) + 3.0
    W = rng.normal(size=(n_features, n_targets))
    y = X @ W + 5.0 + rng.normal(size=(n_samples, n_targets))
    return X, y


def fit_default(X, y, alpha, fit_intercept):
    """Existing Ridge, with alpha baked into the prefit. Its warning for n_features >= n_samples is silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('alpha', ALPHAS)
def test_scalar_alpha_matches_default_ridge(shape, fit_intercept, alpha):
    X, y = make_data(*shape)
    model_default = fit_default(X, y, alpha=alpha, fit_intercept=fit_intercept)
    model_svd = Ridge(alpha=alpha, fit_intercept=fit_intercept, prefit_includes_alpha=False).fit(X, y)

    np.testing.assert_allclose(model_svd.coef_, model_default.coef_, **TOLERANCES)
    np.testing.assert_allclose(model_svd.intercept_, model_default.intercept_, **TOLERANCES)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_per_target_alpha_matches_default_ridge(shape, fit_intercept):
    X, y = make_data(*shape)
    alphas = np.array(ALPHAS)  ## one per target column
    model_svd = Ridge(fit_intercept=fit_intercept, prefit_includes_alpha=False).fit(X, y, alpha=alphas)

    for i_target, alpha in enumerate(alphas):
        model_default = fit_default(X, y[:, i_target], alpha=alpha, fit_intercept=fit_intercept)
        np.testing.assert_allclose(model_svd.coef_[:, i_target], model_default.coef_, **TOLERANCES)
        np.testing.assert_allclose(model_svd.intercept_[i_target], model_default.intercept_, **TOLERANCES)


## Scalar penalties for both prefit modes, and per-target penalties for the SVD
## mode. The large penalties are where a penalized intercept would be shrunk
## toward zero.
ALPHAS_LARGE = [1e5, 1e7]
MODES_AND_ALPHAS_SKLEARN = (
    [(True, alpha) for alpha in [*ALPHAS, *ALPHAS_LARGE]]
    + [(False, alpha) for alpha in [*ALPHAS, *ALPHAS_LARGE]]
    + [(False, np.array([1e-2, 1.0, 1e5, 1e7]))]  ## one per target column
)

@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('use_prefit', [False, True])
@pytest.mark.parametrize('prefit_includes_alpha, alpha', MODES_AND_ALPHAS_SKLEARN)
def test_matches_sklearn(shape, fit_intercept, use_prefit, prefit_includes_alpha, alpha):
    """The intercept is not penalized, as in sklearn's Ridge."""
    X, y = make_data(*shape)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  ## the default mode warns for n_features >= n_samples
        model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            prefit_X=X if use_prefit else None,
            prefit_includes_alpha=prefit_includes_alpha,
        ).fit(X, y)
    model_sk = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept, solver='svd').fit(X, y)

    np.testing.assert_allclose(model.coef_, model_sk.coef_.T, **TOLERANCES)
    np.testing.assert_allclose(model.intercept_, model_sk.intercept_, **TOLERANCES)


@pytest.mark.parametrize('prefit_includes_alpha', [True, False])
def test_large_alpha_intercept_is_mean_y(prefit_includes_alpha):
    """As alpha grows, coef_ goes to zero and intercept_ goes to mean(y)."""
    X, y = make_data(100, 20)
    model = Ridge(alpha=1e12, prefit_includes_alpha=prefit_includes_alpha).fit(X, y)

    np.testing.assert_allclose(model.coef_, 0, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, y.mean(0), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_one_prefit_reused_across_alphas(shape, fit_intercept):
    X, y = make_data(*shape)
    model_prefit = Ridge(fit_intercept=fit_intercept, prefit_X=X, prefit_includes_alpha=False)
    svd_X = model_prefit.svd_X

    alphas_all = [*ALPHAS, np.array(ALPHAS), np.array(ALPHAS[::-1])]
    for alpha in alphas_all:
        model_prefit.fit(X, y, alpha=alpha)
        model_fresh = Ridge(alpha=alpha, fit_intercept=fit_intercept, prefit_includes_alpha=False).fit(X, y)
        np.testing.assert_allclose(model_prefit.coef_, model_fresh.coef_, **TOLERANCES)
        np.testing.assert_allclose(model_prefit.intercept_, model_fresh.intercept_, **TOLERANCES)
    assert model_prefit.svd_X is svd_X, 'fit must reuse the prefit, not replace it'
    assert model_prefit.iXTXaeXT is None


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_1d_y_shapes(fit_intercept):
    X, y = make_data(100, 20, n_targets=1)
    y = y[:, 0]
    model_default = fit_default(X, y, alpha=10.0, fit_intercept=fit_intercept)
    model_svd = Ridge(alpha=10.0, fit_intercept=fit_intercept, prefit_includes_alpha=False).fit(X, y)

    assert model_svd.coef_.shape == (20,)
    assert np.ndim(model_svd.intercept_) == 0
    np.testing.assert_allclose(model_svd.coef_, model_default.coef_, **TOLERANCES)
    np.testing.assert_allclose(model_svd.intercept_, model_default.intercept_, **TOLERANCES)


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_torch_matches_numpy(device, fit_intercept):
    X, y = make_data(40, 150)
    alphas = np.array(ALPHAS)
    model_np = Ridge(fit_intercept=fit_intercept, prefit_X=X, prefit_includes_alpha=False).fit(X, y, alpha=alphas)

    X_t, y_t = (torch.as_tensor(arr, dtype=torch.float64, device=device) for arr in (X, y))
    model_t = Ridge(fit_intercept=fit_intercept, prefit_X=X_t, prefit_includes_alpha=False)
    model_t.fit(X_t, y_t, alpha=torch.as_tensor(alphas))  ## alpha on cpu is moved to X's device

    assert isinstance(model_t.coef_, torch.Tensor) and model_t.coef_.device.type == device
    np.testing.assert_allclose(model_t.coef_.cpu().numpy(), model_np.coef_, **TOLERANCES)
    np.testing.assert_allclose(model_t.intercept_.cpu().numpy(), model_np.intercept_, **TOLERANCES)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('model_class, kwargs', [
    (Ridge, dict(alpha=10.0)),
    (ReducedRankRidgeRegression, dict(rank=2, alpha=10.0)),
])
def test_torch_matches_numpy_alpha_in_prefit(device, fit_intercept, model_class, kwargs):
    """Ridge's default prefit mode and ReducedRankRidgeRegression, with the prefit and X means on the device."""
    X, y = make_data(40, 150)
    X_t, y_t = (torch.as_tensor(arr, dtype=torch.float64, device=device) for arr in (X, y))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  ## Ridge's default mode warns for n_features >= n_samples
        model_np = model_class(fit_intercept=fit_intercept, prefit_X=X, **kwargs).fit(X, y)
        model_t = model_class(fit_intercept=fit_intercept, prefit_X=X_t, **kwargs).fit(X_t, y_t)

    assert isinstance(model_t.coef_, torch.Tensor) and model_t.coef_.device.type == device
    np.testing.assert_allclose(model_t.coef_.cpu().numpy(), model_np.coef_, **TOLERANCES)
    np.testing.assert_allclose(model_t.intercept_.cpu().numpy(), model_np.intercept_, **TOLERANCES)


def test_errors():
    X, y = make_data(100, 20)
    with pytest.raises(AssertionError, match='prefit_includes_alpha=False'):
        Ridge(alpha=1.0).fit(X, y, alpha=10.0)
    with pytest.raises(AssertionError, match='one value per target'):
        Ridge(prefit_includes_alpha=False).fit(X, y, alpha=np.ones(N_TARGETS + 1))


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('use_prefit', [False, True])
@pytest.mark.parametrize('alpha', [1.0, 1e5])
@pytest.mark.parametrize('rank', [1, 2, N_TARGETS])
def test_reduced_rank_matches_sklearn(shape, fit_intercept, use_prefit, alpha, rank):
    """
    The coefficients are sklearn's Ridge coefficients projected onto their top
    ``rank`` right singular vectors, and the intercept is not penalized. At
    full rank this is sklearn's Ridge.
    """
    X, y = make_data(*shape)
    model = ReducedRankRidgeRegression(
        rank=rank,
        alpha=alpha,
        fit_intercept=fit_intercept,
        prefit_X=X if use_prefit else None,
    ).fit(X, y)

    model_sk = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept, solver='svd').fit(X, y)
    coef_sk = model_sk.coef_.T  ## (n_features, n_targets)
    Vt = np.linalg.svd(coef_sk, full_matrices=False)[2][:rank]  ## (rank, n_targets)
    coef_sk_reduced = coef_sk @ Vt.T @ Vt  ## (n_features, n_targets)
    np.testing.assert_allclose(model.coef_, coef_sk_reduced, **TOLERANCES)

    if not fit_intercept:
        np.testing.assert_allclose(model.intercept_, 0, **TOLERANCES)
    elif rank == N_TARGETS:
        np.testing.assert_allclose(model.intercept_, model_sk.intercept_, **TOLERANCES)
    else:
        np.testing.assert_allclose(model.intercept_, y.mean(0) - X.mean(0) @ coef_sk_reduced, **TOLERANCES)
