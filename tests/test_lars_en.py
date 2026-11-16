"""
Tests for lars_elastic_net_path, the exact elastic-net path for a fixed L2
penalty by least angle regression with the lasso modification. All references
are scikit-learn, in float64:
- lambda2 = 0: ``lars_path(X, y, method='lasso')``.
- lambda2 > 0: ``lars_path_gram`` with Gram = X^T X + n*lambda2*I, which is the
  exact elastic-net path for small n_features.
- The optimality conditions at every knot, and ``ElasticNet`` between knots.

Penalty convention: (1/(2n))||y - Xb||^2 + lambda1*||b||_1 + (lambda2/2)*||b||^2.
The largest differences are printed; run pytest with -s to see them.
"""

import warnings

import numpy as np
import pytest
import sklearn.datasets
import sklearn.exceptions
import sklearn.linear_model
import torch

from torch_linear_regression import lars_elastic_net_path, interpolate_lars_path, SingularGramError


LAMBDA2S_POSITIVE = [0.01, 0.1, 1.0, 10.0]
RTOL_SKLEARN_LARS = 1e-10  ## knots and coefficients against sklearn's LARS
TOL_OPTIMALITY = 1e-10  ## relative to lambda1 at the same point (at lambda1 = 0, relative to lambda1 at the first knot)
RTOL_ELASTIC_NET = 1e-8  ## limited by coordinate descent's tolerance


## Data ##

def make_data(n_samples, n_features, seed, rho=0.0):
    """
    Centered X whose neighboring columns have AR(1) correlation rho, and a
    centered y from a sparse linear model plus noise.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_samples, n_features))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    for j in range(1, n_features):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * Z[:, j]
    beta = np.zeros(n_features)
    n_true = min(8, n_features)
    beta[rng.choice(n_features, size=n_true, replace=False)] = rng.normal(size=n_true)
    y = X @ beta + rng.normal(size=n_samples)
    return X - X.mean(axis=0), y - y.mean()


def make_duplicate_data(noise_sd, seed=0):
    """Columns 20-39 are columns 0-19 plus Gaussian noise of SD noise_sd (0 gives exact copies)."""
    rng = np.random.default_rng(seed)
    X_half = rng.normal(size=(60, 20))
    X = np.concatenate([X_half, X_half + noise_sd * rng.normal(size=X_half.shape)], axis=1)
    y = X_half[:, :6] @ rng.normal(size=6) + rng.normal(size=60)
    return X - X.mean(axis=0), y - y.mean()


def load_diabetes():
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    return X - X.mean(axis=0), y - y.mean()


## name: (function that returns X and y, whether n_features > n_samples)
CASES = {
    'diabetes': (load_diabetes, False),
    **{f'p<n_seed{seed}': (lambda seed=seed: make_data(60, 15, seed), False) for seed in range(3)},
    **{f'p>n_seed{seed}': (lambda seed=seed: make_data(40, 120, seed), True) for seed in range(3)},
    **{f'ar1_p<n_seed{seed}': (lambda seed=seed: make_data(60, 15, seed, rho=0.9), False) for seed in range(2)},
    **{f'ar1_p>n_seed{seed}': (lambda seed=seed: make_data(40, 120, seed, rho=0.9), True) for seed in range(2)},
}


def get_case(name, lambda2):
    """X, y and max_steps. With p > n, lambda2 = 0 stops well before n features are active, where the active Gram becomes singular."""
    make, is_wide = CASES[name]
    X, y = make()
    if not is_wide:
        max_steps = 500  ## full path, down to lambda1 = 0
    else:
        max_steps = 25 if lambda2 == 0 else 60
    return X, y, max_steps


## Helpers ##

def fit_ours(X, y, lambda2, **kwargs):
    return lars_elastic_net_path(X=torch.as_tensor(X), y=torch.as_tensor(y), lambda2=lambda2, **kwargs)


def densify(path):
    """Coefficients at every knot. Shape: (n_knots, n_features)."""
    coefs = np.zeros((len(path['knots_index']), path['n_features']))
    for k, (index, coef) in enumerate(zip(path['knots_index'], path['knots_coef'])):
        coefs[k, index.numpy()] = coef.numpy()
    return coefs


def events_ours(path):
    return list(zip(path['events_index'].tolist(), path['events_type'].tolist()))


def fit_sklearn(X, y, lambda2, max_steps, lambda1_min=0.0):
    """sklearn's lasso-LARS path. Returns lambda1 at the knots (n_knots,), coefficients (n_knots, n_features), and the final active list."""
    n_samples, n_features = X.shape
    if lambda2 == 0:
        alphas, active, coefs = sklearn.linear_model.lars_path(
            X, y, method='lasso', max_iter=max_steps, alpha_min=lambda1_min,
        )
    else:
        alphas, active, coefs = sklearn.linear_model.lars_path_gram(
            Xy=X.T @ y, Gram=X.T @ X + n_samples * lambda2 * np.eye(n_features), n_samples=n_samples,
            method='lasso', max_iter=max_steps, alpha_min=lambda1_min,
        )
    return alphas, coefs.T, list(active)


def events_sklearn(X, y, lambda2, n_steps):
    """
    sklearn's event sequence, from its final active list after 1, 2, ..., n_steps
    steps. sklearn adds a feature at the start of a step and removes a dropped
    feature at the end of the step in which its coefficient reaches zero. In our
    convention the drop is the event of the next step.
    """
    events = {}  ## step number: (feature, +1 entered or -1 dropped)
    active_prev = []
    for k in range(1, n_steps + 1):
        _, _, active = fit_sklearn(X, y, lambda2, max_steps=k)
        for j in active:
            if j not in active_prev:
                assert k not in events
                events[k] = (j, 1)
        for j in active_prev:
            if j not in active:
                assert k + 1 not in events
                events[k + 1] = (j, -1)
        active_prev = active
    return [events[k] for k in sorted(events) if k <= n_steps]


def max_rel_diff_per_knot(coefs_a, coefs_b):
    """Largest |difference| at each knot divided by the largest |coefficient| at that knot, then the max over knots."""
    scale = np.maximum(np.abs(coefs_b).max(axis=1), np.finfo(float).tiny)
    return float((np.abs(coefs_a - coefs_b).max(axis=1) / scale).max())


def optimality_violation(X, y, coef, lambda1, lambda2):
    """
    Largest violation of the elastic-net optimality conditions at one point,
    with c = X^T (y - X b) / n - lambda2 * b. Active j: c_j = lambda1 * sign(b_j).
    Inactive j: |c_j| <= lambda1.
    """
    n_samples = X.shape[0]
    corr = X.T @ (y - X @ coef) / n_samples - lambda2 * coef
    is_active = coef != 0
    violation_active = np.abs(corr[is_active] - lambda1 * np.sign(coef[is_active])).max(initial=0.0)
    violation_inactive = (np.abs(corr[~is_active]) - lambda1).max(initial=0.0)
    return max(violation_active, violation_inactive, 0.0)


def check_optimality_along_path(X, y, path):
    """
    Largest optimality violation over all knots and the midpoints between them,
    each divided by lambda1 at that point. A point at lambda1 = 0 (the end of a
    full path) is divided by lambda1 at knot 0 instead.
    """
    lambda1s = path['lambda1s'].numpy()
    coefs = densify(path)
    points = [(lambda1s[k], coefs[k]) for k in range(len(lambda1s))]
    for k in range(len(lambda1s) - 1):
        lambda1_mid = 0.5 * (lambda1s[k] + lambda1s[k + 1])
        points.append((lambda1_mid, interpolate_lars_path(path, lambda1_mid).numpy()))
    return max(
        optimality_violation(X, y, coef, lambda1, path['lambda2']) / (lambda1 if lambda1 > 0 else lambda1s[0])
        for lambda1, coef in points
    )


def compare_with_sklearn(name, lambda2):
    X, y, max_steps = get_case(name, lambda2)
    path = fit_ours(X, y, lambda2, max_steps=max_steps)
    lambda1s_sk, coefs_sk, _ = fit_sklearn(X, y, lambda2, max_steps=max_steps)
    n_steps = len(path['events_index'])

    assert len(lambda1s_sk) == n_steps + 1, f"{len(lambda1s_sk) - 1} sklearn steps vs {n_steps} ours"
    events_sk = events_sklearn(X, y, lambda2, n_steps)
    assert events_ours(path) == events_sk

    lambda1s = path['lambda1s'].numpy()
    coefs = densify(path)
    diff_lambda1 = float(np.abs(lambda1s - lambda1s_sk).max() / lambda1s_sk[0])
    diff_coef = max_rel_diff_per_knot(coefs, coefs_sk)
    n_drops = int((path['events_type'] == -1).sum())
    print(f"\n[sklearn LARS] {name} lambda2={lambda2}: {n_steps} steps, {n_drops} drops, stop={path['stop_reason']}, "
          f"max rel diff lambda1={diff_lambda1:.2e}, coef={diff_coef:.2e}")
    np.testing.assert_allclose(lambda1s, lambda1s_sk, rtol=RTOL_SKLEARN_LARS, atol=RTOL_SKLEARN_LARS * lambda1s_sk[0])
    assert diff_coef <= RTOL_SKLEARN_LARS
    return path


## Tests against sklearn's LARS ##

@pytest.mark.parametrize('name', list(CASES))
def test_lasso_matches_sklearn_lars_path(name):
    compare_with_sklearn(name, lambda2=0.0)


def test_diabetes_lasso_has_a_drop():
    """The diabetes lasso path drops a feature, so the drop logic is exercised against sklearn."""
    path = compare_with_sklearn('diabetes', lambda2=0.0)
    assert (path['events_type'] == -1).any()


@pytest.mark.parametrize('lambda2', LAMBDA2S_POSITIVE)
@pytest.mark.parametrize('name', list(CASES))
def test_elastic_net_matches_sklearn_lars_path_gram(name, lambda2):
    compare_with_sklearn(name, lambda2=lambda2)


## Optimality conditions ##

@pytest.mark.parametrize('lambda2', [0.0, *LAMBDA2S_POSITIVE])
@pytest.mark.parametrize('name', list(CASES))
def test_optimality_conditions(name, lambda2):
    X, y, max_steps = get_case(name, lambda2)
    path = fit_ours(X, y, lambda2, max_steps=max_steps)
    violation = check_optimality_along_path(X, y, path)
    print(f"\n[optimality] {name} lambda2={lambda2}: {len(path['events_index'])} steps, max violation / current lambda1 = {violation:.2e}")
    assert violation <= TOL_OPTIMALITY


## sklearn ElasticNet between knots ##

@pytest.mark.parametrize('lambda2', [0.0, 0.1, 1.0, 10.0])
@pytest.mark.parametrize('name', ['p<n_seed0', 'ar1_p<n_seed0', 'p>n_seed0', 'diabetes'])
def test_interpolation_matches_sklearn_elastic_net(name, lambda2):
    """Coordinate descent at the midpoint between each pair of knots, against the interpolation helper."""
    if lambda2 == 0 and CASES[name][1]:
        pytest.skip("The lasso with p > n is not strictly convex, so coordinate descent need not find the same solution.")
    X, y, max_steps = get_case(name, lambda2)
    path = fit_ours(X, y, lambda2, max_steps=max_steps)
    lambda1s = path['lambda1s'].numpy()
    diffs = []
    for k in range(len(lambda1s) - 1):
        lambda1_mid = 0.5 * (lambda1s[k] + lambda1s[k + 1])
        if lambda1_mid <= 1e-6 * lambda1s[0]:
            continue  ## near lambda1 = 0 the lasso is barely regularized and coordinate descent converges slowly
        coef_ours = interpolate_lars_path(path, lambda1_mid).numpy()
        model = sklearn.linear_model.ElasticNet(
            alpha=lambda1_mid + lambda2, l1_ratio=lambda1_mid / (lambda1_mid + lambda2),
            fit_intercept=False, tol=1e-15, max_iter=1_000_000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', sklearn.exceptions.ConvergenceWarning)
            model.fit(X, y)
        diffs.append(np.abs(coef_ours - model.coef_).max() / np.abs(model.coef_).max())
    print(f"\n[ElasticNet] {name} lambda2={lambda2}: {len(diffs)} midpoints, max rel diff = {max(diffs):.2e}")
    assert max(diffs) <= RTOL_ELASTIC_NET


## Duplicate columns ##

@pytest.mark.parametrize('lambda2', [0.01, 0.1, 1.0])
def test_near_duplicate_columns(lambda2):
    X, y = make_duplicate_data(noise_sd=0.01)
    path = fit_ours(X, y, lambda2)
    violation = check_optimality_along_path(X, y, path)
    print(f"\n[near duplicates] lambda2={lambda2}: {len(path['events_index'])} steps, stop={path['stop_reason']}, "
          f"max violation / current lambda1 = {violation:.2e}")
    assert path['stop_reason'] == 'lambda1_min'
    assert violation <= TOL_OPTIMALITY


@pytest.mark.parametrize('lambda2', [0.01, 0.1, 1.0])
def test_exact_duplicate_columns(lambda2):
    """Exact copies get equal coefficients (Zou & Hastie 2005, Theorem 1), and the copy enters right after the original."""
    X, y = make_duplicate_data(noise_sd=0.0)
    path = fit_ours(X, y, lambda2)
    violation = check_optimality_along_path(X, y, path)
    coefs = densify(path)
    diff_copies = max_rel_diff_per_knot(coefs[:, 20:], coefs[:, :20])

    ## lambda1 where each feature first entered
    lambda1s = path['lambda1s'].numpy()
    lambda1_entry = {}
    for k, (j, kind) in enumerate(events_ours(path)):
        if kind == 1 and j not in lambda1_entry:
            lambda1_entry[j] = lambda1s[k]
    gaps = [abs(lambda1_entry[j] - lambda1_entry[j + 20]) / lambda1s[0] for j in range(20) if j in lambda1_entry]
    print(f"\n[exact duplicates] lambda2={lambda2}: {len(path['events_index'])} steps, "
          f"{int((path['events_type'] == -1).sum())} drops, stop={path['stop_reason']}, "
          f"max violation / current lambda1 = {violation:.2e}, max rel diff between copies = {diff_copies:.2e}, "
          f"max entry gap / lambda1_max = {max(gaps):.2e}")
    assert path['stop_reason'] == 'lambda1_min'
    assert violation <= TOL_OPTIMALITY
    assert diff_copies <= 1e-12
    assert all((j + 20 in lambda1_entry) for j in range(20) if j in lambda1_entry)
    assert max(gaps) == 0.0  ## the copy enters after a step of length exactly 0


@pytest.mark.parametrize('copy_signs', [(1, 1), (1, -1), (1, 1, -1)])
@pytest.mark.parametrize('name', ['p>n_seed0', 'ar1_p>n_seed0'])
def test_copies_of_design_match_original_path(name, copy_signs):
    """
    With lambda2 > 0, the path on [s_1 X, ..., s_m X] (s_c = +1 or -1) at
    m*lambda2 is the path on X at lambda2 with coefficient s_c * b / m in copy c
    (substitute b = sum_c s_c g_c). Each event of the original appears m times
    in a row, once per copy, as one group of simultaneous events at exactly the
    same lambda1, drops included, with sign s_c times the original sign. These
    cases have drops, so this also checks that copies drop out together
    instead of cycling.
    """
    lambda2 = 0.01
    X, y, max_steps = get_case(name, lambda2)
    n_features, n_copies = X.shape[1], len(copy_signs)
    X_copies = np.concatenate([s * X for s in copy_signs], axis=1)
    path = fit_ours(X, y, lambda2, max_steps=max_steps)
    path_copies = fit_ours(X_copies, y, n_copies * lambda2, max_steps=n_copies * max_steps)
    n_steps = len(path['events_index'])
    assert (path['events_type'] == -1).any()
    assert path['events_group'].tolist() == list(range(n_steps))  ## no ties in the original

    ## Events: block k of the copies' path holds event k of the original once per copy, as one group
    events = list(zip(path['events_index'].tolist(), path['events_type'].tolist(), path['events_sign'].tolist()))
    events_copies = list(zip(path_copies['events_index'].tolist(), path_copies['events_type'].tolist(), path_copies['events_sign'].tolist()))
    assert len(events_copies) == n_copies * n_steps
    for k, (j, kind, sign) in enumerate(events):
        block = events_copies[n_copies * k: n_copies * (k + 1)]
        assert sorted(block) == sorted((j + c * n_features, kind, s * sign) for c, s in enumerate(copy_signs))
    ## Copies that enter form one group at exactly the same lambda1. Copies that drop sit within 1e-12 of lambda1 but can
    ## fall into separate groups: the coefficient that a copy is left with after the first copy drops is a few rounding
    ## units, and the step that takes it to zero can lower lambda1 by slightly more than the tie tolerance.
    lambda1s, lambda1s_copies = path['lambda1s'].numpy(), path_copies['lambda1s'].numpy()
    groups_copies = path_copies['events_group'].numpy()
    for k, (_, kind, _) in enumerate(events):
        knots_block = lambda1s_copies[n_copies * k: n_copies * (k + 1)]  ## the knots at which the block's events happen
        if kind == 1:
            assert len(set(groups_copies[n_copies * k: n_copies * (k + 1)].tolist())) == 1
            assert (knots_block == knots_block[0]).all()
        else:
            assert np.abs(knots_block - knots_block[0]).max() <= 1e-12 * lambda1s[0]
        if k > 0:
            assert groups_copies[n_copies * k] > groups_copies[n_copies * k - 1]  ## the original's events are not tied
    diff_lambda1 = np.abs(lambda1s_copies[0::n_copies] - lambda1s).max() / lambda1s[0]

    coefs, coefs_copies = densify(path), densify(path_copies)
    diff_coef = max(max_rel_diff_per_knot(s * coefs_copies[0::n_copies, c * n_features:(c + 1) * n_features], coefs / n_copies)
                    for c, s in enumerate(copy_signs))
    diff_between_copies = max(max_rel_diff_per_knot(s * coefs_copies[1:, c * n_features:(c + 1) * n_features], coefs_copies[1:, :n_features])
                              for c, s in enumerate(copy_signs))
    violation = check_optimality_along_path(X_copies, y, path_copies)
    tol_tie_max = float((path_copies['tolerances_tie'] / path_copies['lambda1s'][:-1]).max())
    print(f"\n[copies] {name} signs={copy_signs}: {n_steps} steps, {int((path['events_type'] == -1).sum())} drops; "
          f"max rel diff lambda1={diff_lambda1:.2e}, coef={diff_coef:.2e}, between copies={diff_between_copies:.2e}, "
          f"max violation / current lambda1 = {violation:.2e}, max tie tolerance / lambda1 = {tol_tie_max:.2e}")
    assert diff_lambda1 <= 1e-12
    assert diff_coef <= RTOL_SKLEARN_LARS
    assert diff_between_copies <= 1e-12
    assert violation <= TOL_OPTIMALITY
    assert tol_tie_max <= 1e-10


def test_singular_gram_error_carries_partial_path():
    """
    With lambda2 = 0, exact duplicate columns and n_features > n_samples make
    the active Gram singular. The error is a ValueError that carries the path
    up to the last knot, identical to a run that stops there.
    """
    X, y = make_data(40, 120, seed=0)
    X_dup = np.concatenate([X, X], axis=1)
    with pytest.raises(SingularGramError) as info:
        fit_ours(X_dup, y, 0.0, max_steps=1000)
    assert isinstance(info.value, ValueError)
    path = info.value.path
    n_steps = len(path['events_index'])
    print(f"\n[singular Gram] raised after {n_steps} steps: {info.value}")
    assert path['stop_reason'] == 'singular_gram'
    assert n_steps >= 1
    assert len(path['lambda1s']) == len(path['knots_index']) == len(path['knots_coef']) == n_steps + 1
    for key in ['events_type', 'events_sign', 'events_group', 'tolerances_tie']:
        assert len(path[key]) == n_steps

    path_short = fit_ours(X_dup, y, 0.0, max_steps=n_steps)
    assert path_short['stop_reason'] == 'max_steps'
    for key in ['lambda1s', 'events_index', 'events_type', 'events_sign', 'events_group', 'tolerances_tie']:
        assert torch.equal(path[key], path_short[key])
    np.testing.assert_array_equal(densify(path), densify(path_short))
    assert check_optimality_along_path(X_dup, y, path) <= TOL_OPTIMALITY


@pytest.mark.parametrize('name, lambda2', [('diabetes', 0.0), ('p>n_seed0', 0.01), ('ar1_p>n_seed0', 0.01)])
def test_events_sign_and_group(name, lambda2):
    """
    Without ties, every event is its own group. An entering feature's sign is
    its coefficient's sign at the next knot; a dropped feature's sign is its
    coefficient's sign at the knot before it reached zero.
    """
    X, y, max_steps = get_case(name, lambda2)
    path = fit_ours(X, y, lambda2, max_steps=max_steps)
    coefs = densify(path)
    events = list(zip(path['events_index'].tolist(), path['events_type'].tolist(), path['events_sign'].tolist()))
    assert any(kind == -1 for _, kind, _ in events)
    assert path['events_group'].tolist() == list(range(len(events)))
    for k, (j, kind, sign) in enumerate(events):  ## the event of step k + 1 happens at knot k
        knot = k + 1 if kind == 1 else k - 1
        assert np.sign(coefs[knot, j]) == sign, f"event {k}: feature {j}, type {kind}, sign {sign}, coef {coefs[knot, j]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device.")
@pytest.mark.parametrize('lambda2', [0.0, 0.01, 1.0])
def test_cuda_matches_cpu(lambda2):
    X, y, max_steps = get_case('ar1_p>n_seed0', lambda2)
    path_cpu = fit_ours(X, y, lambda2, max_steps=max_steps)
    path_cuda = lars_elastic_net_path(X=torch.as_tensor(X, device='cuda'), y=torch.as_tensor(y, device='cuda'), lambda2=lambda2, max_steps=max_steps)
    assert path_cuda['lambda1s'].device.type == 'cuda' and path_cuda['knots_coef'][-1].device.type == 'cuda'
    for key in ['events_index', 'events_type', 'events_sign', 'events_group']:
        assert path_cuda[key].tolist() == path_cpu[key].tolist()
    path_cuda = {**path_cuda, 'knots_index': [i.cpu() for i in path_cuda['knots_index']], 'knots_coef': [c.cpu() for c in path_cuda['knots_coef']]}
    diff_lambda1 = float((path_cuda['lambda1s'].cpu() - path_cpu['lambda1s']).abs().max() / path_cpu['lambda1s'][0])
    diff_coef = max_rel_diff_per_knot(densify(path_cuda), densify(path_cpu))
    print(f"\n[cuda] lambda2={lambda2}: max rel diff lambda1={diff_lambda1:.2e}, coef={diff_coef:.2e}")
    assert diff_lambda1 <= 1e-12
    assert diff_coef <= RTOL_SKLEARN_LARS


## float32 ##

def test_float32_agrees_with_float64():
    X, y = make_data(100, 30, seed=0)
    path64 = fit_ours(X, y, 1.0, max_steps=20)
    path32 = fit_ours(X.astype(np.float32), y.astype(np.float32), 1.0, max_steps=20)
    assert path32['lambda1s'].dtype == torch.float32
    assert events_ours(path32) == events_ours(path64)
    diff_lambda1 = float(np.abs(path32['lambda1s'].numpy() - path64['lambda1s'].numpy()).max() / path64['lambda1s'][0].item())
    diff_coef = max_rel_diff_per_knot(densify(path32), densify(path64))
    print(f"\n[float32] max rel diff lambda1={diff_lambda1:.2e}, coef={diff_coef:.2e}")
    assert diff_lambda1 <= 1e-5
    assert diff_coef <= 1e-4


## Stopping rules, storage, interpolation ##

def test_stopping_rules():
    X, y = make_data(60, 15, seed=0)
    path_full = fit_ours(X, y, 0.1)
    assert path_full['stop_reason'] == 'lambda1_min'
    assert path_full['lambda1s'][-1].item() == 0.0

    path = fit_ours(X, y, 0.1, max_steps=5)
    assert path['stop_reason'] == 'max_steps'
    assert len(path['events_index']) == 5 and len(path['lambda1s']) == 6 and len(path['knots_coef']) == 6
    np.testing.assert_array_equal(path['lambda1s'].numpy(), path_full['lambda1s'].numpy()[:6])

    path = fit_ours(X, y, 0.1, max_active=4)
    assert path['stop_reason'] == 'max_active'
    assert max(len(index) for index in path['knots_index']) == 4

    ## Stop between knots 3 and 4: the last knot is lambda1_min, as in sklearn with alpha_min
    lambda1_min = float(0.5 * (path_full['lambda1s'][3] + path_full['lambda1s'][4]))
    path = fit_ours(X, y, 0.1, lambda1_min=lambda1_min)
    assert path['stop_reason'] == 'lambda1_min'
    assert path['lambda1s'][-1].item() == lambda1_min
    assert len(path['events_index']) == 4
    np.testing.assert_allclose(densify(path)[-1], interpolate_lars_path(path_full, lambda1_min).numpy(), rtol=1e-12, atol=1e-14)
    lambda1s_sk, coefs_sk, _ = fit_sklearn(X, y, 0.1, max_steps=500, lambda1_min=lambda1_min)
    np.testing.assert_allclose(path['lambda1s'].numpy(), lambda1s_sk, rtol=RTOL_SKLEARN_LARS)
    assert max_rel_diff_per_knot(densify(path), coefs_sk) <= RTOL_SKLEARN_LARS


def test_storage_is_per_knot_active_set():
    X, y = make_data(40, 120, seed=0)
    path = fit_ours(X, y, 0.1, max_steps=30)
    for index, coef in zip(path['knots_index'], path['knots_coef']):
        assert index.dtype == torch.int64 and coef.shape == index.shape
        assert len(set(index.tolist())) == len(index)
    assert len(path['knots_index'][0]) == 0


def test_interpolation_at_knots_and_outside():
    X, y = make_data(60, 15, seed=1)
    path = fit_ours(X, y, 0.1, max_steps=8)
    coefs = densify(path)
    lambda1s = path['lambda1s'].numpy()
    for k in range(len(lambda1s)):
        np.testing.assert_allclose(interpolate_lars_path(path, lambda1s[k]).numpy(), coefs[k], rtol=1e-14, atol=0)
    assert not interpolate_lars_path(path, 2 * lambda1s[0]).any()
    with pytest.raises(ValueError):
        interpolate_lars_path(path, 0.5 * lambda1s[-1])


def test_numpy_and_torch_inputs_agree():
    X, y = make_data(60, 15, seed=2)
    path_numpy = lars_elastic_net_path(X=X, y=y, lambda2=0.1, max_steps=6)
    path_torch = fit_ours(X, y, 0.1, max_steps=6)
    assert events_ours(path_numpy) == events_ours(path_torch)
    np.testing.assert_array_equal(densify(path_numpy), densify(path_torch))


def test_input_validation():
    X, y = make_data(20, 5, seed=0)
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    with pytest.raises(ValueError):
        lars_elastic_net_path(X=X[0], y=y, lambda2=0.1)  ## 1D X
    with pytest.raises(ValueError):
        lars_elastic_net_path(X=X, y=y[:-1], lambda2=0.1)  ## length mismatch
    with pytest.raises(ValueError):
        lars_elastic_net_path(X=X, y=y[:, None], lambda2=0.1)  ## 2D y
    with pytest.raises(TypeError):
        lars_elastic_net_path(X=X.to(torch.int64), y=y.to(torch.int64), lambda2=0.1)
    with pytest.raises(TypeError):
        lars_elastic_net_path(X=X, y=y.to(torch.float32), lambda2=0.1)
    for value in [float('nan'), float('inf'), -float('inf')]:
        for row, col in [(0, 0), (10, 2), (-1, -1)]:  ## first, middle and last element
            X_bad = X.clone()
            X_bad[row, col] = value
            with pytest.raises(ValueError):
                lars_elastic_net_path(X=X_bad, y=y, lambda2=0.1)
    X_large = torch.ones(64, 4099, dtype=torch.float64)  ## large enough for vectorized and multithreaded reductions
    for index_flat in [0, 64 * 4099 // 2 + 3, 64 * 4099 - 1]:
        X_bad = X_large.clone().reshape(-1)
        X_bad[index_flat] = float('nan')
        with pytest.raises(ValueError):
            lars_elastic_net_path(X=X_bad.reshape(64, 4099), y=torch.ones(64, dtype=torch.float64), lambda2=0.1)
    y_inf = y.clone()
    y_inf[0] = float('inf')
    with pytest.raises(ValueError):
        lars_elastic_net_path(X=X, y=y_inf, lambda2=0.1)
    for kwargs in [dict(lambda2=-1.0), dict(lambda2=float('nan')), dict(lambda2=0.1, max_steps=0),
                   dict(lambda2=0.1, max_active=0), dict(lambda2=0.1, lambda1_min=-1.0)]:
        with pytest.raises(ValueError):
            lars_elastic_net_path(X=X, y=y, **kwargs)
