"""
Ridge regression with marginal maximum likelihood (MML) lambda selection.

Port of ridgeMML.m (Matt Kaufman, 2018) -- Karabatsos 2017 algorithm.
PyTorch backend for SVD, NLL sweep, and beta solve.
Vectorized golden-section search replaces scipy Brent refinement.

X columns are intrinsically z-scored (divide by std, mean-center) before
regression. Coefficients are un-z-scored before being stored, so
``predict(X_raw)`` works on raw (non-z-scored) data.

Reference:
    Karabatsos, G. (2017). Marginal maximum likelihood estimation methods for
    the tuning parameters of ridge, power ridge, and generalized ridge
    regression. Communications in Statistics -- Simulation and Computation.
    http://www.tandfonline.com/doi/pdf/10.1080/03610918.2017.1321119
"""

from typing import Optional, Tuple, Union
import math
import warnings

import numpy as np
import torch

from .linear_regression import LinearRegression_sk


###############################################################################
## Module-level helpers (private)
###############################################################################


def _select_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve device string to ``torch.device``. If ``None``, auto-detect
    cuda / cpu.

    Args:
        device (Optional[Union[str, torch.device]]):
            Device specification. ``None`` means auto-detect.

    Returns:
        (torch.device):
            Resolved device.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device)


def _nll_batch(
    L: torch.Tensor,
    d2: torch.Tensor,
    alpha2: torch.Tensor,
    n: int,
    Y_var: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """
    Batched negative log-likelihood (Eq. 19 of Karabatsos 2017).

    Evaluates the NLL at per-column lambda values ``L``.

    Args:
        L (torch.Tensor):
            Per-column lambda values. Shape: ``(p_y,)``.
        d2 (torch.Tensor):
            Squared singular values (truncated to q). Shape: ``(q,)``.
        alpha2 (torch.Tensor):
            Squared alpha values (truncated to q). Shape: ``(q, p_y)``.
        n (int):
            Number of observations.
        Y_var (torch.Tensor):
            Column-wise sum of squared (centered) Y. Shape: ``(p_y,)``.
        q (int):
            Number of valid singular values.

    Returns:
        (torch.Tensor):
            NLL values, shape ``(p_y,)``. Invalid entries are ``inf``.
    """
    ## Ld2: (q, p_y) -- broadcast d2 (q,) with L (p_y,)
    Ld2 = L[None, :] + d2[:, None]  ## (q, p_y)

    ## ratio_sum: sum over q of alpha2 / Ld2 -> (p_y,)
    ratio_sum = torch.sum(alpha2 / Ld2, dim=0)  ## (p_y,)

    ## inner = Y_var - ratio_sum; clamp before log to avoid log(negative)
    inner = Y_var - ratio_sum  ## (p_y,)
    inner_safe = inner.clamp(min=1e-300)  ## (p_y,)

    ## log(Ld2) summed over q -> (p_y,)
    log_Ld2_sum = torch.sum(torch.log(Ld2), dim=0)  ## (p_y,)

    ## L must be > 0 for log(L) to be valid; clamp for safety
    L_safe = L.clamp(min=1e-300)  ## (p_y,)

    ## NLL = -(q*log(L) - sum(log(L+d2)) - n*log(inner))
    nll = -(q * torch.log(L_safe) - log_Ld2_sum - n * torch.log(inner_safe))  ## (p_y,)

    ## Mask invalid entries: inner <= 0 or L <= 0
    inf_tensor = torch.tensor(float("inf"), dtype=nll.dtype, device=nll.device)
    nll = torch.where((inner > 0) & (L > 0), nll, inf_tensor)  ## (p_y,)

    return nll


def _vectorized_nll_sweep(
    q: int,
    d2: torch.Tensor,
    alpha2: torch.Tensor,
    n: int,
    Y_var: torch.Tensor,
    p_y: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized two-phase NLL sweep over all ``p_y`` columns simultaneously.

    Phase 1: step size 1/4 for lambda in ``[0, step_switch]``.
    Phase 2: adaptive step size ``(L / step_denom)`` with boxcar(7) smoothing.

    Returns per-column ``(min_L, max_L)`` bounds for golden-section refinement.

    Args:
        q (int):
            Number of valid singular values.
        d2 (torch.Tensor):
            Squared singular values (truncated to q). Shape: ``(q,)``.
        alpha2 (torch.Tensor):
            Squared alpha values (truncated to q). Shape: ``(q, p_y)``.
        n (int):
            Number of observations.
        Y_var (torch.Tensor):
            Sum of squared Y column values. Shape: ``(p_y,)``.
        p_y (int):
            Number of Y columns.
        device (torch.device):
            Computation device.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]):
            bounds (torch.Tensor):
                Shape ``(p_y, 2)``. Column 0 = min_L, column 1 = max_L.
            hit_cap (torch.Tensor):
                Boolean tensor, shape ``(p_y,)``. True if the lambda cap was
                hit during the sweep (convergence failure).
    """
    ## Constants -- matching MATLAB exactly
    smooth = 7
    step_switch = 25
    step_denom = 100
    max_lambda_cap = 1e8

    ## Pre-allocate scalar constants outside loop
    _dtype = torch.float64
    inf_val = torch.tensor(float("inf"), dtype=_dtype, device=device)

    ## Output buffers
    min_L = torch.zeros(p_y, dtype=_dtype, device=device)
    max_L = torch.zeros(p_y, dtype=_dtype, device=device)
    done_mask = torch.zeros(p_y, dtype=torch.bool, device=device)
    hit_cap = torch.zeros(p_y, dtype=torch.bool, device=device)

    ## Rolling smoothing buffer: (smooth, p_y)
    sm_buffer = torch.full((smooth, p_y), float("nan"), dtype=_dtype, device=device)
    test_vals_L = torch.full((smooth, p_y), float("nan"), dtype=_dtype, device=device)
    sm_buffer_i = -1  ## will be incremented before use

    ## Vectorized NLL computation helper (scalar L, broadcast across p_y)
    def _compute_nll_scalar(L_val: float) -> torch.Tensor:
        """Compute NLL for scalar L across all p_y columns. Returns (p_y,)."""
        if L_val <= 0:
            return torch.full((p_y,), float("inf"), dtype=_dtype, device=device)
        Ld2 = L_val + d2  ## (q,)
        ## sum(alpha2 / Ld2) over q dimension: (q, p_y) / (q, 1) -> sum -> (p_y,)
        ratio_sum = torch.sum(alpha2 / Ld2[:, None], dim=0)  ## (p_y,)
        inner = Y_var - ratio_sum  ## (p_y,)
        inner_safe = inner.clamp(min=1e-300)  ## (p_y,)
        log_Ld2_sum = torch.sum(torch.log(Ld2))  ## scalar (same for all columns)

        nll = -(q * math.log(L_val) - log_Ld2_sum - n * torch.log(inner_safe))  ## (p_y,)
        ## Mask invalid entries
        nll = torch.where(inner > 0, nll, inf_val)  ## (p_y,)
        return nll

    ## Phase 1: step size 1/4, lambda from 0 to stepSwitch
    prev_NLL = torch.full((p_y,), float("inf"), dtype=_dtype, device=device)

    for k in range(int(step_switch * 4) + 1):
        sm_buffer_i = (sm_buffer_i + 1) % smooth
        L_val = k / 4.0

        NLL = _compute_nll_scalar(L_val)  ## (p_y,)

        ## Store in rolling buffer
        sm_buffer[sm_buffer_i, :] = NLL
        test_vals_L[sm_buffer_i, :] = L_val

        ## Check which columns just passed their minimum (NLL > prev_NLL and not yet done)
        just_passed = (NLL > prev_NLL) & (~done_mask)
        if just_passed.any():
            val_min = torch.tensor((k - 2) / 4.0, dtype=_dtype, device=device)
            val_max = torch.tensor(k / 4.0, dtype=_dtype, device=device)
            min_L = torch.where(just_passed, val_min, min_L)
            max_L = torch.where(just_passed, val_max, max_L)
            done_mask = done_mask | just_passed

        prev_NLL = torch.where(done_mask, prev_NLL, NLL)

        if done_mask.all():
            break

    ## Phase 2: adaptive step with smoothing for columns not yet done
    if not done_mask.all():
        L_current = k / 4.0  ## last k from phase 1

        ## Reset prev_NLL for undone columns to the buffer mean
        prev_NLL_phase2 = torch.nanmean(sm_buffer, dim=0)  ## (p_y,)
        prev_NLL = torch.where(done_mask, prev_NLL, prev_NLL_phase2)

        while not done_mask.all():
            L_current = L_current + L_current / step_denom
            sm_buffer_i = (sm_buffer_i + 1) % smooth

            NLL_new = _compute_nll_scalar(L_current)  ## (p_y,)

            ## Update buffer only for undone columns
            sm_buffer[sm_buffer_i, :] = torch.where(
                done_mask, sm_buffer[sm_buffer_i, :], NLL_new,
            )
            L_current_tensor = torch.tensor(L_current, dtype=_dtype, device=device)
            test_vals_L[sm_buffer_i, :] = torch.where(
                done_mask, test_vals_L[sm_buffer_i, :], L_current_tensor,
            )

            ## Smoothed NLL for undone columns
            NLL_smooth = torch.nanmean(sm_buffer, dim=0)  ## (p_y,)

            ## Check which undone columns passed minimum
            just_passed = (NLL_smooth > prev_NLL) & (~done_mask)
            if just_passed.any():
                ## Walk back by half kernel: (smooth-1)/2 = 3 steps back from current
                walk_back = (smooth - 1) // 2  ## = 3
                idx_max = (sm_buffer_i - walk_back) % smooth
                idx_min = (idx_max - 2) % smooth

                ## Extract max_L and min_L from the buffer for these columns
                max_L = torch.where(just_passed, test_vals_L[idx_max, :], max_L)
                min_L = torch.where(just_passed, test_vals_L[idx_min, :], min_L)
                done_mask = done_mask | just_passed

            ## Update prev_NLL for undone columns
            prev_NLL = torch.where(done_mask, prev_NLL, NLL_smooth)

            ## Safety cap
            if L_current > max_lambda_cap:
                still_undone = ~done_mask
                if still_undone.any():
                    warnings.warn(
                        f"RidgeMML lambda search hit cap (L={L_current:.2e} > "
                        f"{max_lambda_cap:.2e}). NLL landscape is flat -- using "
                        f"cap as upper bound for {still_undone.sum().item()} "
                        f"columns.",
                        stacklevel=3,
                    )
                    val_min_cap = torch.tensor(max_lambda_cap / 2.0, dtype=_dtype, device=device)
                    val_max_cap = torch.tensor(max_lambda_cap, dtype=_dtype, device=device)
                    min_L = torch.where(still_undone, val_min_cap, min_L)
                    max_L = torch.where(still_undone, val_max_cap, max_L)
                    hit_cap = hit_cap | still_undone
                    done_mask = torch.ones(p_y, dtype=torch.bool, device=device)

    ## Pack bounds as (p_y, 2) tensor
    bounds = torch.stack([min_L, max_L], dim=1)  ## (p_y, 2)

    return bounds, hit_cap


def _vectorized_golden_section(
    d2: torch.Tensor,
    alpha2: torch.Tensor,
    n: int,
    Y_var: torch.Tensor,
    q: int,
    bounds: torch.Tensor,
    xatol: float = 1e-4,
    max_iter: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized golden-section search for the NLL minimum over all ``p_y``
    columns simultaneously.

    One NLL evaluation per iteration (reuses the other point from the previous
    iteration). Replaces the sequential scipy Brent loop.

    Args:
        d2 (torch.Tensor):
            Squared singular values (truncated to q). Shape: ``(q,)``.
        alpha2 (torch.Tensor):
            Squared alpha values (truncated to q). Shape: ``(q, p_y)``.
        n (int):
            Number of observations.
        Y_var (torch.Tensor):
            Sum of squared Y column values. Shape: ``(p_y,)``.
        q (int):
            Number of valid singular values.
        bounds (torch.Tensor):
            Per-column ``(min_L, max_L)`` from the sweep. Shape: ``(p_y, 2)``.
        xatol (float):
            Absolute tolerance on the bracket width.
        max_iter (int):
            Maximum number of iterations.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]):
            lambdas_opt (torch.Tensor):
                Optimal lambda per column, shape ``(p_y,)``.
            did_not_converge (torch.Tensor):
                Boolean, shape ``(p_y,)``. True where the bracket width
                exceeds ``xatol`` after ``max_iter`` iterations.
    """
    ## Golden ratio constants
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0   ## ~0.6180
    inv_phi2 = 1.0 - inv_phi                  ## ~0.3820

    _dtype = bounds.dtype
    _device = bounds.device
    p_y = bounds.shape[0]

    ## Clamp lower bounds to >= 0
    a = bounds[:, 0].clamp(min=0.0)  ## (p_y,)
    b = bounds[:, 1].clone()         ## (p_y,)

    ## Initial interior points
    c = a + inv_phi2 * (b - a)  ## (p_y,)
    d = a + inv_phi * (b - a)   ## (p_y,)

    ## Evaluate NLL at both interior points (2 evals for init)
    fc = _nll_batch(L=c, d2=d2, alpha2=alpha2, n=n, Y_var=Y_var, q=q)  ## (p_y,)
    fd = _nll_batch(L=d, d2=d2, alpha2=alpha2, n=n, Y_var=Y_var, q=q)  ## (p_y,)

    for _ in range(max_iter):
        ## Check convergence
        width = b - a  ## (p_y,)
        if (width < xatol).all():
            break

        ## Branch mask: True where fc < fd (shrink right), False (shrink left)
        branch_left = fc < fd  ## (p_y,)

        ## Update brackets
        ## branch_left:  b <- d, d <- c, fd <- fc, new c, eval fc
        ## branch_right: a <- c, c <- d, fc <- fd, new d, eval fd
        a_new = torch.where(branch_left, a, c)   ## (p_y,)
        b_new = torch.where(branch_left, d, b)   ## (p_y,)

        ## After updating a,b: the kept interior point.
        ## Both branches select the same source value (old c or old d),
        ## but they serve distinct semantic roles: c_kept becomes the new d
        ## for branch_left columns, and d_kept becomes the new c for
        ## branch_right columns. Separated for clarity at assembly below.
        c_kept = torch.where(branch_left, c, d)   ## interior point that stays
        fc_kept = torch.where(branch_left, fc, fd)

        d_kept = torch.where(branch_left, c, d)   ## same value, distinct role per branch
        fd_kept = torch.where(branch_left, fc, fd)

        ## Compute the single new eval point per column
        ## branch_left:  new c = a_new + inv_phi2 * (b_new - a_new)
        ## branch_right: new d = a_new + inv_phi  * (b_new - a_new)
        new_c_candidate = a_new + inv_phi2 * (b_new - a_new)  ## (p_y,)
        new_d_candidate = a_new + inv_phi * (b_new - a_new)   ## (p_y,)

        ## The point to evaluate: for branch_left it's new_c, for branch_right it's new_d
        eval_point = torch.where(branch_left, new_c_candidate, new_d_candidate)  ## (p_y,)

        ## Single batched NLL evaluation
        f_eval = _nll_batch(L=eval_point, d2=d2, alpha2=alpha2, n=n, Y_var=Y_var, q=q)  ## (p_y,)

        ## Assemble new state
        ## branch_left:  c=new_c_candidate, fc=f_eval, d=old_c, fd=old_fc
        ## branch_right: c=old_d, fc=old_fd, d=new_d_candidate, fd=f_eval
        a = a_new
        b = b_new
        c = torch.where(branch_left, new_c_candidate, d_kept)
        fc = torch.where(branch_left, f_eval, fd_kept)
        d = torch.where(branch_left, c_kept, new_d_candidate)
        fd = torch.where(branch_left, fc_kept, f_eval)

    ## Optimal lambda: midpoint of final bracket
    lambdas_opt = (a + b) / 2.0  ## (p_y,)

    ## Convergence check
    did_not_converge = (b - a) >= xatol  ## (p_y,)

    return lambdas_opt, did_not_converge


def _batched_solve(
    XTX: torch.Tensor,
    ep: torch.Tensor,
    lambdas: torch.Tensor,
    XTY: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Solve ``(XTX + lambda_i * ep) @ beta_i = XTY[:, i]`` for all columns
    simultaneously.

    Batches the leading dimension (p_y) to control memory usage.

    Args:
        XTX (torch.Tensor):
            ``X^T X`` matrix. Shape: ``(p, p)``.
        ep (torch.Tensor):
            Penalty identity matrix. Shape: ``(p, p)``.
        lambdas (torch.Tensor):
            Per-column ridge parameters. Shape: ``(p_y,)``.
        XTY (torch.Tensor):
            ``X^T Y`` matrix. Shape: ``(p, p_y)``.
        batch_size (Optional[int]):
            Number of columns to solve at once. ``None`` means all.

    Returns:
        (torch.Tensor):
            Beta coefficients. Shape: ``(p, p_y)``.
    """
    p_y = lambdas.shape[0]
    p = XTX.shape[0]

    if batch_size is None:
        batch_size = p_y

    betas = torch.empty(p, p_y, dtype=XTX.dtype, device=XTX.device)

    for start in range(0, p_y, batch_size):
        end = min(start + batch_size, p_y)

        ## Build batch of (XTX + lambda_i * ep) matrices
        ## A shape: (bs, p, p)
        A = XTX[None, :, :] + lambdas[start:end, None, None] * ep[None, :, :]

        ## RHS: (bs, p, 1) -> solve -> (bs, p, 1) -> squeeze
        rhs = XTY[:, start:end].T[:, :, None]  ## (bs, p, 1)

        ## torch.linalg.solve: A @ x = rhs
        x = torch.linalg.solve(A, rhs)  ## (bs, p, 1)
        betas[:, start:end] = x[..., 0].T  ## (p, bs)

    return betas


###############################################################################
## RidgeMML class
###############################################################################


class RidgeMML(LinearRegression_sk):
    """
    Ridge regression with per-column marginal maximum likelihood lambda
    estimation (Karabatsos 2017).

    X columns are intrinsically z-scored (divided by ``std(ddof=1)``,
    mean-centered) before regression. Coefficients are un-z-scored before
    being stored, so ``predict(X_raw)`` works on raw data.

    The intercept is derived as ``Y_mean - X_mean_raw @ coef_`` after
    computing betas on centered data.

    RH 2025

    Args:
        fit_intercept (bool):
            Must be ``True``. ``False`` raises ``NotImplementedError`` at
            fit time.
        lambdas (Optional[np.ndarray]):
            Pre-supplied per-column ridge parameters, shape ``(p_y,)``.
            If ``None`` (default), lambdas are estimated via MML.
        device (Optional[Union[str, torch.device]]):
            Torch device for computation. ``None`` means auto-detect
            (cuda if available, else cpu).
        batch_size_solve (Optional[int]):
            Number of Y columns to solve simultaneously in the beta solve.
            ``None`` (default) means all at once. Reduce for large ``p_y``
            to limit GPU memory.

    Attributes:
        coef_ (Union[np.ndarray, torch.Tensor]):
            Regression coefficients, shape ``(p_x, p_y)``.
        intercept_ (Union[np.ndarray, torch.Tensor]):
            Intercept, shape ``(p_y,)``.
        lambdas_ (Union[np.ndarray, torch.Tensor]):
            Estimated (or pre-supplied) ridge parameters, shape ``(p_y,)``.
        convergence_failures_ (Union[np.ndarray, torch.Tensor]):
            Boolean, shape ``(p_y,)``. True where the lambda search did not
            converge.
        n_features_in_ (int):
            Number of features seen during ``fit``.

    Example::

        import numpy as np
        from torch_linear_regression import RidgeMML

        X = np.random.randn(500, 20)
        beta_true = np.random.randn(20, 5)
        Y = X @ beta_true + 0.5 * np.random.randn(500, 5)

        model = RidgeMML()
        model.fit(X, Y)
        Y_pred = model.predict(X)
        print(model.score(X, Y))
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        lambdas: Optional[np.ndarray] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size_solve: Optional[int] = None,
    ):
        super(RidgeMML, self).__init__()
        self.fit_intercept = fit_intercept
        self.lambdas = lambdas
        self.device = device
        self.batch_size_solve = batch_size_solve

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "RidgeMML":
        """
        Fit ridge regression with MML lambda estimation.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Design matrix (no intercept column). Shape: ``(n, p_x)``.
                Must be 2-D.
            y (Union[np.ndarray, torch.Tensor]):
                Outcome matrix. Shape: ``(n, p_y)``. Must be 2-D.

        Returns:
            (RidgeMML):
                Fitted estimator (``self``).

        Raises:
            NotImplementedError:
                If ``fit_intercept`` is ``False``.
            ValueError:
                If shapes are incompatible or inputs are not 2-D.
        """
        ## -----------------------------------------------------------
        ## Guard: fit_intercept=False not supported
        ## -----------------------------------------------------------
        if not self.fit_intercept:
            raise NotImplementedError(
                "RidgeMML only supports fit_intercept=True. "
                "The no-intercept path is not implemented."
            )

        ## -----------------------------------------------------------
        ## Input validation
        ## -----------------------------------------------------------
        input_is_torch = isinstance(X, torch.Tensor)
        input_device = X.device if input_is_torch else None

        ## Convert to numpy float64 for preprocessing
        if input_is_torch:
            X_np = X.detach().cpu().numpy().astype(np.float64)
            y_np = y.detach().cpu().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
            y_np = np.asarray(y, dtype=np.float64)

        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got ndim={X_np.ndim}")
        if y_np.ndim == 1:
            y_np = y_np[:, None]  ## (n,) -> (n, 1)
        if y_np.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D, got ndim={y_np.ndim}")
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X_np.shape[0]} rows, "
                f"y has {y_np.shape[0]} rows"
            )

        n = X_np.shape[0]
        p_x = X_np.shape[1]
        p_y = y_np.shape[1]
        self.n_features_in_ = p_x

        dev = _select_device(self.device)

        ## -----------------------------------------------------------
        ## Store raw X column means (before any transformation)
        ## -----------------------------------------------------------
        X_mean_raw = X_np.mean(axis=0)  ## (p_x,)

        ## -----------------------------------------------------------
        ## Mean-center Y
        ## -----------------------------------------------------------
        Y_mean = y_np.mean(axis=0)  ## (p_y,)
        Y_centered = y_np - Y_mean[None, :]  ## (n, p_y)

        ## -----------------------------------------------------------
        ## Z-score X: divide by std(ddof=1), then mean-center
        ## -----------------------------------------------------------
        X_std = np.std(X_np, axis=0, ddof=1)  ## (p_x,)
        X_std[X_std == 0] = 1.0  ## prevent division by zero
        X_zscored = X_np / X_std[None, :]  ## (n, p_x)

        X_mean_zscored = X_zscored.mean(axis=0)  ## (p_x,)
        X_centered = X_zscored - X_mean_zscored[None, :]  ## (n, p_x)

        ## -----------------------------------------------------------
        ## Determine whether to compute lambdas
        ## -----------------------------------------------------------
        compute_lambdas = True
        lambdas_precomputed = None
        if self.lambdas is not None:
            lambdas_precomputed = np.asarray(self.lambdas, dtype=np.float64)
            if not np.isnan(lambdas_precomputed[0]):
                compute_lambdas = False

        ## Validate pre-supplied lambdas shape
        if lambdas_precomputed is not None and not np.isnan(lambdas_precomputed[0]):
            if len(lambdas_precomputed) != p_y:
                raise ValueError(
                    f"lambdas has length {len(lambdas_precomputed)}, "
                    f"expected {p_y} (number of y columns)"
                )

        ## Single-regressor edge case: skip optimization, set lambda=1
        if p_x == 1 and compute_lambdas:
            compute_lambdas = False
            lambdas_precomputed = np.ones(p_y)

        ## -----------------------------------------------------------
        ## Lambda optimization
        ## -----------------------------------------------------------
        convergence_failures = np.zeros(p_y, dtype=bool)

        if compute_lambdas:
            ## Move to torch for SVD and NLL sweep
            X_t = torch.as_tensor(X_centered, dtype=torch.float64, device=dev)
            Y_t = torch.as_tensor(Y_centered, dtype=torch.float64, device=dev)

            ## SVD of X_centered
            U_t, s_t, Vh_t = torch.linalg.svd(X_t, full_matrices=False)  ## U:(n,p), s:(p,), Vh:(p,p)

            ## Number of valid singular values (MATLAB line 131)
            eps_u1_np = np.spacing(U_t[0, 0].cpu().item())
            threshold_t = eps_u1_np * torch.arange(1, p_x + 1, dtype=torch.float64, device=dev)
            q = int((s_t > threshold_t).sum().item())

            d2_t = s_t ** 2  ## (p,)

            ## alpha = S * U' * Y (MATLAB line 139)
            alph_t = s_t[:, None] * (U_t.T @ Y_t)  ## (p, p_y)
            alpha2_t = alph_t ** 2  ## (p, p_y)

            ## Variance of Y columns: sum(Y.^2, 1)
            Y_var_t = torch.sum(Y_t ** 2, dim=0)  ## (p_y,)

            ## Phase 1 + Phase 2: vectorized NLL sweep for bounds
            bounds_t, hit_cap_t = _vectorized_nll_sweep(
                q=q,
                d2=d2_t[:q],
                alpha2=alpha2_t[:q, :],
                n=n,
                Y_var=Y_var_t,
                p_y=p_y,
                device=dev,
            )

            ## Phase 3: vectorized golden-section refinement (replaces scipy Brent)
            lambdas_opt_t, did_not_converge_t = _vectorized_golden_section(
                d2=d2_t[:q],
                alpha2=alpha2_t[:q, :],
                n=n,
                Y_var=Y_var_t,
                q=q,
                bounds=bounds_t,
                xatol=1e-4,
                max_iter=200,
            )

            ## Propagate convergence failures: hit_cap OR golden-section didn't converge
            convergence_failures_t = hit_cap_t | did_not_converge_t  ## (p_y,)

            lambdas_final = lambdas_opt_t.cpu().numpy()  ## (p_y,)
            convergence_failures = convergence_failures_t.cpu().numpy()

            ## Free GPU memory
            del X_t, Y_t, U_t, s_t, Vh_t, d2_t, alph_t, alpha2_t, Y_var_t
            del bounds_t, hit_cap_t, lambdas_opt_t, did_not_converge_t, convergence_failures_t
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        else:
            lambdas_final = lambdas_precomputed

        ## -----------------------------------------------------------
        ## Beta solve via _batched_solve (recenter=True path only)
        ## -----------------------------------------------------------
        X_t = torch.as_tensor(X_centered, dtype=torch.float64, device=dev)
        Y_t = torch.as_tensor(Y_centered, dtype=torch.float64, device=dev)
        lambdas_t = torch.as_tensor(lambdas_final, dtype=torch.float64, device=dev)

        XTX_t = X_t.T @ X_t  ## (p_x, p_x)
        ep_t = torch.eye(p_x, dtype=torch.float64, device=dev)  ## (p_x, p_x)
        XTY_t = X_t.T @ Y_t  ## (p_x, p_y)

        betas_t = _batched_solve(
            XTX=XTX_t,
            ep=ep_t,
            lambdas=lambdas_t,
            XTY=XTY_t,
            batch_size=self.batch_size_solve,
        )  ## (p_x, p_y)

        ## Undo z-scoring: betas /= X_std
        X_std_t = torch.as_tensor(X_std, dtype=torch.float64, device=dev)  ## (p_x,)
        betas_t = betas_t / X_std_t[:, None]  ## (p_x, p_y)

        ## -----------------------------------------------------------
        ## Compute intercept: Y_mean - X_mean_raw @ coef_
        ## -----------------------------------------------------------
        X_mean_raw_t = torch.as_tensor(X_mean_raw, dtype=torch.float64, device=dev)  ## (p_x,)
        Y_mean_t = torch.as_tensor(Y_mean, dtype=torch.float64, device=dev)  ## (p_y,)
        intercept_t = Y_mean_t - X_mean_raw_t @ betas_t  ## (p_y,)

        ## -----------------------------------------------------------
        ## Convert outputs to numpy
        ## -----------------------------------------------------------
        coef_np = betas_t.cpu().numpy()  ## (p_x, p_y)
        intercept_np = intercept_t.cpu().numpy()  ## (p_y,)

        ## Free GPU memory
        del X_t, Y_t, lambdas_t, XTX_t, ep_t, XTY_t, betas_t
        del X_std_t, X_mean_raw_t, Y_mean_t, intercept_t
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        ## -----------------------------------------------------------
        ## Store results, matching input type (np or torch)
        ## -----------------------------------------------------------
        if input_is_torch:
            self.coef_ = torch.as_tensor(coef_np, device=input_device)
            self.intercept_ = torch.as_tensor(intercept_np, device=input_device)
            self.lambdas_ = torch.as_tensor(lambdas_final, device=input_device)
            self.convergence_failures_ = torch.as_tensor(convergence_failures, device=input_device)
        else:
            self.coef_ = coef_np
            self.intercept_ = intercept_np
            self.lambdas_ = lambdas_final
            self.convergence_failures_ = convergence_failures

        return self

    ## -----------------------------------------------------------
    ## Override to/cpu/numpy to handle lambdas_ and convergence_failures_
    ## -----------------------------------------------------------

    def to(self, device):
        """Move all fitted attributes to the given device."""
        self.coef_ = torch.as_tensor(self.coef_, device=device)
        self.intercept_ = torch.as_tensor(self.intercept_, device=device)
        if hasattr(self, "lambdas_"):
            self.lambdas_ = torch.as_tensor(self.lambdas_, device=device)
        if hasattr(self, "convergence_failures_"):
            self.convergence_failures_ = torch.as_tensor(self.convergence_failures_, device=device)
        return self

    def cpu(self):
        """Move all fitted attributes to CPU."""
        return self.to("cpu")

    def numpy(self):
        """Convert all fitted attributes to numpy arrays."""
        def _convert(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        self.coef_ = _convert(self.coef_)
        self.intercept_ = _convert(self.intercept_)
        if hasattr(self, "lambdas_"):
            self.lambdas_ = _convert(self.lambdas_)
        if hasattr(self, "convergence_failures_"):
            self.convergence_failures_ = _convert(self.convergence_failures_)
        return self
