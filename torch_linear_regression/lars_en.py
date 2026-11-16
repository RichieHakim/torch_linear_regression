"""
Exact elastic-net regularization path for a fixed L2 penalty, computed by least
angle regression with the lasso modification (LARS-EN).
References:
    Zou H, Hastie T (2005). Regularization and variable selection via the
    elastic net. JRSS-B 67(2):301-320. Lemma 1 and Section 3.4.
    Efron B, Hastie T, Johnstone I, Tibshirani R (2004). Least angle
    regression. Annals of Statistics 32(2):407-499. Section 3.1.
RH 2026
"""

from typing import Dict, Any, Optional, Union, List

import numpy as np
import torch


N_ULPS_TIE = 8  ## smallest tie tolerance, in rounding units of n*lambda1


class SingularGramError(ValueError):
    """
    Raised by ``lars_elastic_net_path`` when the active-set Gram is numerically
    singular. Carries the path computed up to that point. RH 2026

    Attributes:
        path (Dict[str, Any]):
            The path up to the last knot before the failure, in the format
            returned by ``lars_elastic_net_path``, with ``stop_reason =
            'singular_gram'``. The event that made the Gram singular is not
            included; its feature is named in the message.
    """
    def __init__(self, message: str, path: Dict[str, Any]):
        super().__init__(message)
        self.path = path


@torch.no_grad()
def lars_elastic_net_path(
    X: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    lambda2: float,
    max_steps: int = 500,
    max_active: Optional[int] = None,
    lambda1_min: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes the exact elastic-net path for one target: the L2 penalty is fixed
    and the L1 penalty decreases from the value where the first feature enters.
    Records every step, in order: which feature entered or dropped, with which
    sign, and the L1 penalty at each knot. RH 2026

    Penalty convention:
        minimize over b:
            (1/(2n)) * ||y - X b||^2 + lambda1 * ||b||_1 + (lambda2/2) * ||b||^2

        This is a lasso on the augmented data X* = [X; sqrt(n*lambda2) * I],
        y* = [y; 0] with the same 1/(2n) scaling (Zou & Hastie 2005, Lemma 1).
        It matches sklearn's ``ElasticNet(alpha, l1_ratio)`` with
        ``alpha * l1_ratio = lambda1`` and ``alpha * (1 - l1_ratio) = lambda2``.
        sklearn's ``Ridge(alpha=alpha_ridge)`` corresponds to
        ``lambda2 = alpha_ridge / n``. With ``lambda2 = 0`` the knots match
        ``sklearn.linear_model.lars_path(X, y, method='lasso')``, whose
        ``alphas`` are lambda1.

        The returned coefficients are the plain ("naive") elastic-net
        coefficients that sklearn's ``ElasticNet`` returns, not Zou & Hastie's
        rescaled (1 + lambda2) * b.

        At a knot, lambda1 = max_j |c_j| / n, where c_j = X_j^T r - n*lambda2*b_j
        and r = y - X b. For an active feature, c_j = n*lambda1*sign(b_j); for an
        inactive feature, |c_j| <= n*lambda1.

    No centering or scaling is done inside, and there is no intercept. Center
    y and the columns of X (and scale the columns, if wanted) before calling.

    Use float64 when the order of events matters. In float32, two features
    whose entries are within about 1e-4 of lambda1 of each other can enter in
    the opposite order: at 1,000 x 349,000, 2 of 40 float32 paths entered a
    different feature than float64 at step 47 or 48.

    Algorithm and cost:
        Each step factors the active-set Gram G_A = X_A^T X_A + n*lambda2*I
        (n_active x n_active) from scratch, then makes ONE product with X:
        X^T @ [r, u] (n_features x n_samples times n_samples x 2), where u =
        X_A w is the direction of the fitted values. This gives fresh
        correlations X^T r at every step, so they do not drift, and their rates
        of change X^T u for the step lengths. The (n_features x n_features)
        Gram is never formed. Memory is O(n_samples * n_active + n_features) on
        top of X.

    Ties and simultaneous events:
        One event per step. Candidate events that cannot be told apart from
        the current knot at working precision are treated as ties and happen
        after a step of length exactly 0, at exactly the same lambda1. The tie
        tolerance of each step is the larger of N_ULPS_TIE rounding units of
        n*lambda1 and the spread of the active |c_j| below n*lambda1 (in exact
        arithmetic they are all equal, so their spread measures the rounding
        at that step). A candidate is tied if its |c_j| is within the
        tolerance of n*lambda1, or if its step would lower n*lambda1 by no more
        than the tolerance. So a second exact copy (or negated copy) of an
        active column enters right after it, at the same lambda1. Copies that
        drop out can land in separate groups: after the first copy drops, the
        others can keep a coefficient of a few rounding units, and the step
        that takes it to zero can lower lambda1 by slightly more than the
        tolerance (about 1e-14 of lambda1 in tests).

        'events_group' marks simultaneous events: it increases by 1 only when
        lambda1 changes between two events, so events with equal values happen
        at the same lambda1. The order of events within a group is arbitrary
        (it is set by rounding or by column index) and should not be
        interpreted. 'tolerances_tie' returns the tolerance used at each step.

        A feature that just dropped cannot re-enter with its old sign on the
        very next step, where rounding could otherwise make it re-enter at the
        same knot. It can re-enter with the opposite sign.

        With lambda2 = 0, at most n_samples features can be active, and
        duplicate or collinear active columns make the active Gram singular,
        which raises SingularGramError. With lambda2 > 0, the path is unique
        and exact duplicate columns get equal coefficients.

    Steps, knots, and events:
        Step k (k = 1..K) moves from knot k-1 to knot k. Knot 0 is lambda1_max,
        where all coefficients are zero. The event of step k happens at knot k-1,
        at the start of the step: a feature enters, or a feature whose
        coefficient reached zero at knot k-1 drops out.

    Args:
        X (Union[torch.Tensor, np.ndarray]):
            Features. Shape: (n_samples, n_features). float32 or float64.
            Any device.
        y (Union[torch.Tensor, np.ndarray]):
            One target. Shape: (n_samples,). Same type, dtype and device as X.
        lambda2 (float):
            Fixed L2 penalty, >= 0, in the convention above.
        max_steps (int):
            Maximum number of steps (events).
        max_active (Optional[int]):
            Maximum number of active features. The path stops at the knot
            where an entry would exceed this number; that feature is not
            entered. ``None`` means no limit.
        lambda1_min (float):
            Smallest lambda1. The last step is shortened so that the path ends
            exactly at lambda1_min.

    Returns:
        (Dict[str, Any]):
            path (Dict[str, Any]):
                'lambda1s' (torch.Tensor):
                    lambda1 at knots 0..K. Shape: (K+1,). Non-increasing.
                'events_index' (torch.Tensor):
                    Feature index of the event of each step. Shape: (K,).
                    int64.
                'events_type' (torch.Tensor):
                    +1 if the feature entered, -1 if it dropped. Shape: (K,).
                    int64.
                'events_sign' (torch.Tensor):
                    Sign of the feature's coefficient: for an entry, the sign
                    it enters with; for a drop, the sign it had before it
                    reached zero. Shape: (K,). int64, +1 or -1.
                'events_group' (torch.Tensor):
                    Simultaneous-event marker. 0 for the first event; increases
                    by 1 only when lambda1 changes between two events, so
                    events with equal values happen at the same lambda1 (within
                    the tie tolerance). Shape: (K,). int64.
                'tolerances_tie' (torch.Tensor):
                    Tie tolerance of each step, in lambda1 units. Shape: (K,).
                'knots_index' (List[torch.Tensor]):
                    K+1 int64 tensors. Knot k holds the features active during
                    step k, in entry order (knot 0 is empty). A feature that
                    drops at knot k is listed at knot k with coefficient
                    exactly 0.
                'knots_coef' (List[torch.Tensor]):
                    K+1 tensors of coefficients matching 'knots_index'.
                'n_features' (int):
                    Number of features.
                'lambda2' (float):
                    The fixed L2 penalty.
                'stop_reason' (str):
                    'max_steps', 'max_active', or 'lambda1_min' (which includes
                    reaching lambda1 = 0 at the end of the full path). A path
                    carried by SingularGramError has 'singular_gram'.

    Raises:
        SingularGramError:
            If the active-set Gram is numerically singular, which happens with
            duplicate or collinear columns when lambda2 = 0, or with p > n at
            lambda2 = 0 once the active set reaches the rank of X. A subclass
            of ValueError; its ``path`` attribute holds the path computed up
            to that point.
    """
    ## Validate inputs
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {tuple(X.shape)}.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1D with length n_samples={X.shape[0]}. Got shape {tuple(y.shape)}.")
    if X.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"X must be float32 or float64. Got {X.dtype}.")
    if y.dtype != X.dtype:
        raise TypeError(f"y must have the same dtype as X ({X.dtype}). Got {y.dtype}.")
    if y.device != X.device:
        raise ValueError(f"y must be on the same device as X ({X.device}). Got {y.device}.")
    ## min and max propagate NaN and show +-inf, without the (n_samples, n_features) temporary of torch.isfinite(X)
    X_min, X_max = torch.aminmax(X)
    if not (bool(torch.isfinite(X_min)) and bool(torch.isfinite(X_max))):
        raise ValueError("X contains NaN or infinite values.")
    if not bool(torch.isfinite(y).all()):
        raise ValueError("y contains NaN or infinite values.")
    lambda2 = float(lambda2)
    lambda1_min = float(lambda1_min)
    if not (np.isfinite(lambda2) and lambda2 >= 0):
        raise ValueError(f"lambda2 must be finite and >= 0. Got {lambda2}.")
    if not (np.isfinite(lambda1_min) and lambda1_min >= 0):
        raise ValueError(f"lambda1_min must be finite and >= 0. Got {lambda1_min}.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError(f"max_steps must be an integer >= 1. Got {max_steps}.")
    if max_active is not None and (isinstance(max_active, bool) or not isinstance(max_active, (int, np.integer)) or max_active < 1):
        raise ValueError(f"max_active must be None or an integer >= 1. Got {max_active}.")

    n_samples, n_features = X.shape
    dtype, device = X.dtype, X.device
    eps = torch.finfo(dtype).eps
    ridge = n_samples * lambda2  ## added to the diagonal of G_A; also the weight of b_j in an active feature's correlation
    n_active_limit = n_features if lambda2 > 0 else min(n_samples, n_features)  ## with lambda2 = 0, more than n_samples columns are collinear
    inf = float('inf')

    ## Knot 0: all coefficients are zero and lambda1 = max_j |X_j^T y| / n
    corr = X.T @ y  ## (n_features,)
    index_next = int(torch.argmax(corr.abs()))
    lambda1s: List[float] = [float(corr[index_next].abs()) / n_samples]
    type_next = 1  ## +1: enter, -1: drop
    sign_next = 1.0 if float(corr[index_next]) > 0 else -1.0
    is_tied_next = False  ## True if the next event happens at the same lambda1 as the previous one

    index_active: List[int] = []  ## in entry order
    sign_active: List[float] = []
    coef_active = torch.zeros(0, dtype=dtype, device=device)  ## (n_active,)
    knots_index = [torch.zeros(0, dtype=torch.int64, device=device)]
    knots_coef = [torch.zeros(0, dtype=dtype, device=device)]
    events_index: List[int] = []
    events_type: List[int] = []
    events_sign: List[int] = []
    events_group: List[int] = []
    tolerances_tie: List[float] = []

    def make_path(stop_reason: str) -> Dict[str, Any]:
        return {
            'lambda1s': torch.as_tensor(lambda1s, dtype=dtype, device=device),
            'events_index': torch.as_tensor(events_index, dtype=torch.int64, device=device),
            'events_type': torch.as_tensor(events_type, dtype=torch.int64, device=device),
            'events_sign': torch.as_tensor(events_sign, dtype=torch.int64, device=device),
            'events_group': torch.as_tensor(events_group, dtype=torch.int64, device=device),
            'tolerances_tie': torch.as_tensor(tolerances_tie, dtype=dtype, device=device),
            'knots_index': list(knots_index),
            'knots_coef': list(knots_coef),
            'n_features': n_features,
            'lambda2': lambda2,
            'stop_reason': stop_reason,
        }

    stop_reason = 'max_steps'
    if lambda1s[0] <= lambda1_min:
        return make_path('lambda1_min')

    for i_step in range(max_steps):
        ## Apply the event found at the end of the previous step
        if type_next == 1:
            if max_active is not None and len(index_active) >= max_active:
                stop_reason = 'max_active'
                break
            index_active_new = index_active + [index_next]
            sign_active_new = sign_active + [sign_next]
            coef_active_new = torch.cat([coef_active, torch.zeros(1, dtype=dtype, device=device)])
        else:
            pos = index_active.index(index_next)
            index_active_new = index_active[:pos] + index_active[pos + 1:]
            sign_active_new = sign_active[:pos] + sign_active[pos + 1:]
            coef_active_new = torch.cat([coef_active[:pos], coef_active[pos + 1:]])

        ## Equiangular direction on the active set: w = AA * G_A^-1 s_A with AA = (s_A^T G_A^-1 s_A)^(-1/2), so that
        ## every active elastic-net correlation shrinks at the same rate AA (Zou & Hastie 2005, Section 3.4)
        n_active = len(index_active_new)
        idx_active = torch.as_tensor(index_active_new, dtype=torch.int64, device=device)  ## (n_active,)
        sign_active_t = torch.as_tensor(sign_active_new, dtype=dtype, device=device)  ## (n_active,)
        X_active = X[:, idx_active]  ## (n_samples, n_active)
        gram_active = X_active.T @ X_active + ridge * torch.eye(n_active, dtype=dtype, device=device)  ## (n_active, n_active)
        chol, info = torch.linalg.cholesky_ex(gram_active)
        if int(info) != 0 or float(torch.diagonal(chol).min()) ** 2 <= 10 * n_active * eps * float(torch.diagonal(gram_active).max()):
            raise SingularGramError(
                f"The active-set Gram is numerically singular at step {i_step + 1}, when feature {index_next} "
                f"{'enters' if type_next == 1 else 'drops'}, with {n_active} active features. This happens with duplicate or "
                "collinear columns when lambda2 = 0 (or a lambda2 that is tiny for the dtype), or with n_features > n_samples "
                "at lambda2 = 0 once the active set reaches the rank of X. Use lambda2 > 0 or remove the redundant columns. "
                "The path up to the previous knot is in this exception's `path` attribute.",
                path=make_path('singular_gram'),
            )
        w = torch.cholesky_solve(sign_active_t[:, None], chol)[:, 0]  ## (n_active,) G_A^-1 s_A
        AA = 1.0 / float(torch.sqrt(sign_active_t @ w))
        w = AA * w  ## coefficient direction; G_A w = AA * s_A

        ## Record the event
        index_active, sign_active, coef_active = index_active_new, sign_active_new, coef_active_new
        events_index.append(index_next)
        events_type.append(type_next)
        events_sign.append(int(sign_next))
        events_group.append(0 if i_step == 0 else events_group[-1] + (0 if is_tied_next else 1))
        just_entered = (type_next == 1)
        index_just_dropped = index_next if type_next == -1 else None
        sign_just_dropped = sign_next  ## for a drop, the sign of the coefficient before it reached zero

        ## One product with X: correlations with the residual and with the fitted-value direction
        residual = y - X_active @ coef_active  ## (n_samples,)
        u = X_active @ w  ## (n_samples,)
        corr_and_rate = X.T @ torch.stack([residual, u], dim=1)  ## (n_features, 2)
        corr = corr_and_rate[:, 0]  ## X^T r
        rate = corr_and_rate[:, 1]  ## X^T u; an inactive correlation changes as corr - gamma * rate

        ## Elastic-net correlation: an active feature's is X_j^T r - n*lambda2*b_j
        corr_en = corr.index_copy(0, idx_active, corr[idx_active] - ridge * coef_active)  ## (n_features,)
        C = float(corr_en.abs().max())  ## n * lambda1 at the current knot

        ## Tie tolerance, in units of n * lambda1. The active |corr_en| all equal C in exact arithmetic, so their spread below C
        ## is the rounding at this step. A tied entrant's gap is within the tolerance, so it cannot widen the spread beyond it.
        tol_tie = max(N_ULPS_TIE * eps * C, C - float(corr_en[idx_active].abs().min()))
        tolerances_tie.append(tol_tie / n_samples)

        ## Entry: step length at which an inactive |corr_j| reaches the common level C - gamma*AA. A candidate is tied (step of
        ## exactly 0) if its gap to the level, or the drop in the level before it arrives, is within the tie tolerance.
        is_entry_candidate = torch.ones(n_features, dtype=torch.bool, device=device).index_fill(0, idx_active, False)
        if n_active >= n_active_limit:
            is_entry_candidate = torch.zeros_like(is_entry_candidate)
        gap_pos = C - corr  ## (n_features,) distance of corr_j below +C
        gap_neg = C + corr  ## distance of corr_j above -C
        den_pos = AA - rate
        den_neg = AA + rate
        gamma_pos = torch.where(den_pos > 0, gap_pos.clamp(min=0) / den_pos, inf)  ## corr_j reaches +level
        gamma_neg = torch.where(den_neg > 0, gap_neg.clamp(min=0) / den_neg, inf)  ## corr_j reaches -level
        is_pos_side = gamma_pos <= gamma_neg  ## (n_features,) the side reached first, before ties are snapped
        gamma_pos = torch.where((den_pos > 0) & ((gap_pos <= tol_tie) | (gamma_pos * AA <= tol_tie)), 0.0, gamma_pos)
        gamma_neg = torch.where((den_neg > 0) & ((gap_neg <= tol_tie) | (gamma_neg * AA <= tol_tie)), 0.0, gamma_neg)
        if index_just_dropped is not None:
            ## The feature that just dropped sits at the level on its old side and moves inward (Efron et al. 2004),
            ## so it cannot re-enter there in this step; rounding could make it look like a step of 0. It can
            ## still re-enter with the opposite sign.
            if sign_just_dropped > 0:
                gamma_pos[index_just_dropped] = inf
                is_pos_side[index_just_dropped] = False
            else:
                gamma_neg[index_just_dropped] = inf
                is_pos_side[index_just_dropped] = True
        gamma_enter_all = torch.where(is_entry_candidate, torch.minimum(gamma_pos, gamma_neg), inf)  ## (n_features,)
        index_enter = int(torch.argmin(gamma_enter_all))
        gamma_enter = float(gamma_enter_all[index_enter])

        ## Drop: step length at which an active coefficient reaches zero (lasso modification, Efron et al. 2004, Section 3.1).
        ## A coefficient at or past zero that keeps moving toward the wrong sign drops after a step of 0, as does one whose
        ## step would lower the level by no more than the tie tolerance.
        sign_w = sign_active_t * w
        gamma_drop_all = torch.where(sign_w < 0, (sign_active_t * coef_active).clamp(min=0) / (-sign_w), inf)  ## (n_active,)
        gamma_drop_all = torch.where(gamma_drop_all * AA <= tol_tie, 0.0, gamma_drop_all)
        if just_entered:
            gamma_drop_all[-1] = inf  ## the feature that just entered sits at zero
        pos_drop = int(torch.argmin(gamma_drop_all))
        gamma_drop = float(gamma_drop_all[pos_drop])

        ## Step to the nearest event, or to lambda1_min
        gamma_end = max(C - n_samples * lambda1_min, 0.0) / AA
        gamma = min(gamma_enter, gamma_drop, gamma_end)
        coef_active = coef_active + gamma * w
        reached_end = gamma_end <= min(gamma_enter, gamma_drop)
        if reached_end:
            lambda1_knot = lambda1_min
        else:
            if gamma == 0.0:
                lambda1_knot = lambda1s[-1]  ## a tie: the next event happens at the same lambda1
            else:
                lambda1_knot = min(max((C - gamma * AA) / n_samples, lambda1_min), lambda1s[-1])  ## clipped so rounding cannot reorder knots
            is_tied_next = (gamma == 0.0)
            if gamma_drop < gamma_enter:
                coef_active = torch.where(torch.arange(n_active, device=device) == pos_drop, 0.0, coef_active)  ## exactly zero at the drop knot
                type_next, index_next = -1, index_active[pos_drop]
                sign_next = sign_active[pos_drop]
            else:
                type_next, index_next = 1, index_enter
                sign_next = 1.0 if bool(is_pos_side[index_enter]) else -1.0

        lambda1s.append(lambda1_knot)
        knots_index.append(idx_active)
        knots_coef.append(coef_active)
        if reached_end:
            stop_reason = 'lambda1_min'
            break

    return make_path(stop_reason)


def interpolate_lars_path(
    path: Dict[str, Any],
    lambda1: float,
) -> torch.Tensor:
    """
    Returns the coefficients at any lambda1 on a path from
    ``lars_elastic_net_path``, by linear interpolation between the two knots
    around it. For a fixed lambda2 the elastic-net path is piecewise linear in
    lambda1, so this is exact up to rounding. RH 2026

    Args:
        path (Dict[str, Any]):
            Output of ``lars_elastic_net_path``.
        lambda1 (float):
            L1 penalty, in the convention of ``lars_elastic_net_path``. At or
            above the first knot, all coefficients are zero.

    Returns:
        (torch.Tensor):
            coef (torch.Tensor):
                Dense coefficients. Shape: (n_features,).

    Raises:
        ValueError:
            If lambda1 is below the last knot, where the path was not computed.
    """
    lambda1 = float(lambda1)
    lambda1s = path['lambda1s'].tolist()
    if not np.isfinite(lambda1) or lambda1 < lambda1s[-1]:
        raise ValueError(f"lambda1 must be finite and >= the last knot ({lambda1s[-1]}). Got {lambda1}.")

    coef = torch.zeros(path['n_features'], dtype=path['lambda1s'].dtype, device=path['lambda1s'].device)
    i_right = sum(value > lambda1 for value in lambda1s)  ## first knot at or below lambda1 (knots are non-increasing)
    if i_right == 0:
        return coef
    i_left = i_right - 1
    fraction = (lambda1s[i_left] - lambda1) / (lambda1s[i_left] - lambda1s[i_right])  ## in (0, 1]; the denominator is > 0
    ## A feature that enters in this interval is absent from the left knot, where its value is 0
    coef = coef.index_add(0, path['knots_index'][i_left], (1 - fraction) * path['knots_coef'][i_left])
    coef = coef.index_add(0, path['knots_index'][i_right], fraction * path['knots_coef'][i_right])
    return coef
