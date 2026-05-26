"""
Ridge regression with marginal maximum likelihood (MML) lambda selection.

Karabatsos 2017 algorithm with PyTorch acceleration for SVD, vectorized NLL
sweep, batched beta solve, and vectorized golden-section refinement.

X columns are intrinsically z-scored (divide by std, mean-center) before
regression -- this is required for lambda comparability across features.
Coefficients are un-z-scored before storage, so ``predict(X_raw)`` works
on raw data.

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


## Floor for clamping values before log() to avoid log(0) or log(negative).
_LOG_CLAMP_MIN = 1e-300


###############################################################################
## Helpers
###############################################################################


def _select_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve device string to ``torch.device``. ``None`` auto-detects cuda/cpu.

    Args:
        device (Optional[Union[str, torch.device]]):
            Device specification. ``None`` means auto-detect.

    Returns:
        (torch.device):
            Resolved device.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    Batched NLL (Eq. 19 of Karabatsos 2017) for per-column lambda values.

    ``NLL(L) = -(q*log(L) - sum_j(log(L + d_j^2)) - n*log(Y_var - sum_j(alpha_j^2 / (L + d_j^2))))``

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
    ## Broadcast: d2 (q,1) + L (1,p_y) -> Ld2 (q, p_y)
    Ld2 = L[None, :] + d2[:, None]  ## (q, p_y)
    ratio_sum = torch.sum(alpha2 / Ld2, dim=0)  ## (p_y,) -- sum over singular values
    inner = Y_var - ratio_sum  ## (p_y,) -- argument of the outer log
    log_Ld2_sum = torch.sum(torch.log(Ld2), dim=0)  ## (p_y,)

    ## Clamp before log to avoid NaN from log(<=0)
    nll = -(
        q * torch.log(L.clamp(min=_LOG_CLAMP_MIN))
        - log_Ld2_sum
        - n * torch.log(inner.clamp(min=_LOG_CLAMP_MIN))
    )  ## (p_y,)

    ## Mask entries where the NLL is undefined
    valid = (inner > 0) & (L > 0)  ## (p_y,)
    return torch.where(valid, nll, torch.inf)


def _nll_scalar_broadcast(
    L_val: float,
    d2: torch.Tensor,
    alpha2: torch.Tensor,
    n: int,
    q: int,
    Y_var: torch.Tensor,
) -> torch.Tensor:
    """
    NLL for a single scalar lambda broadcast across all ``p_y`` columns.

    Same math as ``_nll_batch`` but exploits the fact that ``log(L + d_j^2)``
    is shared across columns, making it O(q) instead of O(q * p_y) for that
    term.

    Args:
        L_val (float):
            Lambda value (scalar, same for all columns).
        d2 (torch.Tensor):
            Squared singular values. Shape: ``(q,)``.
        alpha2 (torch.Tensor):
            Squared alpha values. Shape: ``(q, p_y)``.
        n (int):
            Number of observations.
        q (int):
            Number of valid singular values.
        Y_var (torch.Tensor):
            Column-wise sum of squared Y. Shape: ``(p_y,)``.

    Returns:
        (torch.Tensor):
            NLL values, shape ``(p_y,)``. Invalid entries are ``inf``.
    """
    if L_val <= 0:
        return torch.full_like(Y_var, float("inf"))

    ## Because L is scalar, Ld2 is (q,) shared across all columns.
    ## This makes log_Ld2_sum a scalar rather than (p_y,) -- O(q) not O(q*p_y).
    Ld2 = L_val + d2  ## (q,)
    ratio_sum = torch.sum(alpha2 / Ld2[:, None], dim=0)  ## (q,p_y) / (q,1) -> sum -> (p_y,)
    inner = Y_var - ratio_sum  ## (p_y,)
    log_Ld2_sum = torch.sum(torch.log(Ld2))  ## scalar (shared across all columns)

    nll = -(
        q * math.log(L_val)
        - log_Ld2_sum
        - n * torch.log(inner.clamp(min=_LOG_CLAMP_MIN))
    )  ## (p_y,)

    return torch.where(inner > 0, nll, torch.inf)


def _vectorized_nll_sweep(
    q: int,
    d2: torch.Tensor,
    alpha2: torch.Tensor,
    n: int,
    Y_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-phase NLL sweep over all ``p_y`` columns simultaneously.

    Phase 1: fixed step of ``1 / STEPS_PER_UNIT`` for lambda in
    ``[0, PHASE1_LIMIT]``.
    Phase 2: adaptive step ``L * (1 + 1/ADAPTIVE_DENOM)`` with boxcar
    smoothing of width ``SMOOTH_KERNEL``.

    Returns per-column bracket bounds for golden-section refinement.

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

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]):
            bounds (torch.Tensor):
                Shape ``(p_y, 2)``. Column 0 = min_L, column 1 = max_L.
            hit_cap (torch.Tensor):
                Boolean, shape ``(p_y,)``. True if the cap was hit
                (convergence failure).
    """
    ## Algorithm constants (Karabatsos 2017, Sec. 3.1)
    SMOOTH_KERNEL = 7        ## boxcar smoothing width for phase-2 NLL
    PHASE1_LIMIT = 25        ## phase 1 scans lambda in [0, PHASE1_LIMIT]
    STEPS_PER_UNIT = 4       ## phase 1 step size = 1 / STEPS_PER_UNIT = 0.25
    ADAPTIVE_DENOM = 100     ## phase 2 growth: L_{k+1} = L_k * (1 + 1/ADAPTIVE_DENOM)
    LAMBDA_CAP = 1e8         ## safety cap for flat NLL landscapes

    p_y = alpha2.shape[1]
    device = d2.device

    ## Output buffers
    min_L = torch.zeros(p_y, dtype=torch.float64, device=device)
    max_L = torch.zeros(p_y, dtype=torch.float64, device=device)
    done_mask = torch.zeros(p_y, dtype=torch.bool, device=device)
    hit_cap = torch.zeros(p_y, dtype=torch.bool, device=device)

    ## Rolling smoothing buffer: (SMOOTH_KERNEL, p_y)
    sm_buffer = torch.full((SMOOTH_KERNEL, p_y), float("nan"), dtype=torch.float64, device=device)
    test_vals_L = torch.full((SMOOTH_KERNEL, p_y), float("nan"), dtype=torch.float64, device=device)
    sm_idx = -1

    ## Phase 1: fixed-step sweep
    prev_NLL = torch.full((p_y,), float("inf"), dtype=torch.float64, device=device)
    n_phase1_steps = int(PHASE1_LIMIT * STEPS_PER_UNIT) + 1

    for k in range(n_phase1_steps):
        sm_idx = (sm_idx + 1) % SMOOTH_KERNEL
        L_val = k / STEPS_PER_UNIT

        NLL = _nll_scalar_broadcast(L_val=L_val, d2=d2, alpha2=alpha2, n=n, q=q, Y_var=Y_var)  ## (p_y,)

        sm_buffer[sm_idx, :] = NLL
        test_vals_L[sm_idx, :] = L_val

        ## Detect columns whose NLL just increased (passed through minimum).
        ## The bracket is [k-2, k] in step units -- one step before and after
        ## the minimum, giving golden-section a tight interval to refine.
        just_passed = (NLL > prev_NLL) & (~done_mask)
        if just_passed.any():
            bracket_lo = (k - 2) / STEPS_PER_UNIT
            bracket_hi = k / STEPS_PER_UNIT
            min_L = torch.where(just_passed, bracket_lo, min_L)
            max_L = torch.where(just_passed, bracket_hi, max_L)
            done_mask = done_mask | just_passed

        ## Freeze prev_NLL for done columns so future comparisons don't trigger
        prev_NLL = torch.where(done_mask, prev_NLL, NLL)

        if done_mask.all():
            break

    ## Phase 2: adaptive step with boxcar smoothing.
    ## Step size grows geometrically: L_{k+1} = L_k * (1 + 1/ADAPTIVE_DENOM).
    ## NLL is smoothed with a rolling boxcar mean to handle noisy landscapes.
    if not done_mask.all():
        L_current = (n_phase1_steps - 1) / STEPS_PER_UNIT

        ## Switch from raw NLL to smoothed NLL as the comparison baseline
        prev_NLL = torch.where(done_mask, prev_NLL, torch.nanmean(sm_buffer, dim=0))

        while not done_mask.all():
            L_current = L_current + L_current / ADAPTIVE_DENOM
            sm_idx = (sm_idx + 1) % SMOOTH_KERNEL

            NLL_new = _nll_scalar_broadcast(L_val=L_current, d2=d2, alpha2=alpha2, n=n, q=q, Y_var=Y_var)

            ## Update buffer only for undone columns
            sm_buffer[sm_idx, :] = torch.where(done_mask, sm_buffer[sm_idx, :], NLL_new)
            test_vals_L[sm_idx, :] = torch.where(done_mask, test_vals_L[sm_idx, :], L_current)

            NLL_smooth = torch.nanmean(sm_buffer, dim=0)  ## (p_y,)

            ## Detect columns that passed minimum in the smoothed signal
            just_passed = (NLL_smooth > prev_NLL) & (~done_mask)
            if just_passed.any():
                ## Walk back by half the kernel width to bracket the true minimum.
                ## The smoothed minimum lags the raw minimum by ~half_kernel steps.
                ## idx_hi/idx_lo give a 2-step-wide bracket centered on the estimate.
                half_kernel = (SMOOTH_KERNEL - 1) // 2  ## = 3 for kernel width 7
                idx_hi = (sm_idx - half_kernel) % SMOOTH_KERNEL
                idx_lo = (idx_hi - 2) % SMOOTH_KERNEL
                max_L = torch.where(just_passed, test_vals_L[idx_hi, :], max_L)
                min_L = torch.where(just_passed, test_vals_L[idx_lo, :], min_L)
                done_mask = done_mask | just_passed

            prev_NLL = torch.where(done_mask, prev_NLL, NLL_smooth)

            ## Safety cap for flat NLL landscapes
            if L_current > LAMBDA_CAP:
                still_undone = ~done_mask
                if still_undone.any():
                    warnings.warn(
                        f"RidgeMML lambda search hit cap ({L_current:.2e} > "
                        f"{LAMBDA_CAP:.2e}). NLL landscape is flat -- using "
                        f"cap as upper bound for {still_undone.sum().item()} "
                        f"columns.",
                        stacklevel=3,
                    )
                    min_L = torch.where(still_undone, LAMBDA_CAP / 2.0, min_L)
                    max_L = torch.where(still_undone, LAMBDA_CAP, max_L)
                    hit_cap = hit_cap | still_undone
                    done_mask = torch.ones_like(done_mask)

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
    max_iter: int = 60,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized golden-section search for the NLL minimum across all ``p_y``
    columns simultaneously.

    One NLL evaluation per iteration (the other interior point is reused
    from the previous iteration).

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
            Per-column ``[min_L, max_L]`` from the sweep. Shape: ``(p_y, 2)``.
        xatol (float):
            Absolute tolerance on the bracket width.
        max_iter (int):
            Maximum iterations. 60 covers all practical bracket widths to
            ``xatol=1e-4``: ``ceil(log(1e-4 / 5e7) / log(0.618)) ~ 56``.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]):
            lambdas_opt (torch.Tensor):
                Optimal lambda per column, shape ``(p_y,)``.
            did_not_converge (torch.Tensor):
                Boolean, shape ``(p_y,)``. True where the bracket width
                exceeds ``xatol`` after ``max_iter`` iterations.
    """
    ## Golden ratio conjugate and its complement
    INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0   ## ~0.6180
    INV_PHI2 = 1.0 - INV_PHI                  ## ~0.3820

    ## Initialize bracket endpoints
    a = bounds[:, 0].clamp(min=0.0)  ## (p_y,)
    b = bounds[:, 1].clone()         ## (p_y,)

    ## Initial interior points (2 evals to bootstrap)
    c = a + INV_PHI2 * (b - a)  ## (p_y,)
    d = a + INV_PHI * (b - a)   ## (p_y,)
    fc = _nll_batch(L=c, d2=d2, alpha2=alpha2, n=n, Y_var=Y_var, q=q)  ## (p_y,)
    fd = _nll_batch(L=d, d2=d2, alpha2=alpha2, n=n, Y_var=Y_var, q=q)  ## (p_y,)

    for _ in range(max_iter):
        if ((b - a) < xatol).all():
            break

        ## Per-column branch: shrink_right where fc < fd, else shrink_left
        shrink_right = fc < fd  ## (p_y,)

        ## Update bracket endpoints
        a_new = torch.where(shrink_right, a, c)  ## (p_y,)
        b_new = torch.where(shrink_right, d, b)  ## (p_y,)

        ## The interior point that survives (reused, not re-evaluated)
        kept = torch.where(shrink_right, c, d)      ## (p_y,)
        f_kept = torch.where(shrink_right, fc, fd)   ## (p_y,)

        ## The single new point to evaluate
        ## shrink_right: new c at INV_PHI2 position; shrink_left: new d at INV_PHI position
        new_point = torch.where(
            shrink_right,
            a_new + INV_PHI2 * (b_new - a_new),
            a_new + INV_PHI * (b_new - a_new),
        )  ## (p_y,)
        f_new = _nll_batch(L=new_point, d2=d2, alpha2=alpha2, n=n, Y_var=Y_var, q=q)

        ## Reassemble state
        ## shrink_right: c=new_point, fc=f_new, d=kept(old c), fd=f_kept(old fc)
        ## shrink_left:  c=kept(old d), fc=f_kept(old fd), d=new_point, fd=f_new
        a = a_new
        b = b_new
        c = torch.where(shrink_right, new_point, kept)
        fc = torch.where(shrink_right, f_new, f_kept)
        d = torch.where(shrink_right, kept, new_point)
        fd = torch.where(shrink_right, f_kept, f_new)

    lambdas_opt = (a + b) / 2.0  ## (p_y,)
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
    Solve ``(XTX + lambda_i * ep) @ beta_i = XTY[:, i]`` for all columns.

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
            Columns to solve at once. ``None`` means all.

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

        ## Broadcast: XTX (1,p,p) + lambdas (bs,1,1) * I (1,p,p) -> A (bs,p,p)
        A = XTX[None, :, :] + lambdas[start:end, None, None] * ep[None, :, :]  ## (bs, p, p)
        ## Reshape XTY columns to (bs, p, 1) for batched solve
        rhs = XTY[:, start:end].T[:, :, None]  ## (bs, p, 1)
        x = torch.linalg.solve(A, rhs)  ## (bs, p, 1)
        betas[:, start:end] = x[..., 0].T  ## drop last dim and transpose back to (p, bs)

    return betas


###############################################################################
## RidgeMML
###############################################################################


class RidgeMML(LinearRegression_sk):
    """
    Ridge regression with per-column marginal maximum likelihood lambda
    estimation (Karabatsos 2017).

    Key differences from ``Ridge``:

    - **Per-column lambda.** Each column of Y gets its own regularization
      parameter, estimated via marginal maximum likelihood.
    - **Intrinsic z-scoring.** X columns are always standardized internally
      (divided by ``std(ddof=1)``, mean-centered). This is required by the
      algorithm for lambda comparability. Coefficients are un-z-scored
      before storage, so ``predict(X_raw)`` works on raw data.

    RH 2025

    Args:
        fit_intercept (bool):
            Must be ``True``. ``False`` raises ``NotImplementedError``.
        lambdas (Optional[np.ndarray]):
            Pre-supplied per-column ridge parameters, shape ``(p_y,)``.
            If ``None`` (default), lambdas are estimated via MML.
        device (Optional[Union[str, torch.device]]):
            Torch device for computation. ``None`` auto-detects cuda/cpu.
        batch_size_solve (Optional[int]):
            Y columns to solve simultaneously in the beta solve.
            ``None`` (default) means all at once. Reduce for large ``p_y``
            to limit GPU memory.
        prefit_X (Optional[Union[np.ndarray, torch.Tensor]]):
            If provided, precomputes X preprocessing (z-scoring, centering)
            and SVD at construction time. Subsequent ``fit()`` calls reuse
            these cached results, skipping the most expensive X-only work.
            Useful when fitting the same X to many different Y matrices.

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

        ## Prefit: cache X preprocessing + SVD for repeated fits
        model_pf = RidgeMML(prefit_X=X)
        model_pf.fit(X, Y[:, :3])   ## fast -- reuses cached SVD
        model_pf.fit(X, Y[:, 3:])   ## fast again
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        lambdas: Optional[np.ndarray] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size_solve: Optional[int] = None,
        prefit_X: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super(RidgeMML, self).__init__()
        self.fit_intercept = fit_intercept
        self.lambdas = lambdas
        self.device = device
        self.batch_size_solve = batch_size_solve

        self._prefit_data = self.prefit(prefit_X) if prefit_X is not None else None

    def prefit(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> dict:
        """
        Precompute X preprocessing (z-scoring, centering) and SVD.

        The returned dict is stored internally and reused by ``fit()`` to
        skip these steps. This is the most expensive Y-independent work;
        caching it is useful when fitting the same X to many different Y
        matrices.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Design matrix. Shape: ``(n, p_x)``.

        Returns:
            (dict):
                Cached preprocessing and SVD results (numpy arrays).
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got ndim={X_np.ndim}")

        dev = _select_device(self.device)
        n, p_x = X_np.shape

        ## Z-score and mean-center X (ddof=1 for sample std)
        X_mean_raw = X_np.mean(axis=0)  ## (p_x,)
        X_std = np.std(X_np, axis=0, ddof=1)  ## (p_x,)
        X_std[X_std == 0] = 1.0
        X_zscored = X_np / X_std[None, :]  ## (n, p_x)
        X_centered = X_zscored - X_zscored.mean(axis=0)[None, :]  ## (n, p_x)

        ## SVD of centered, z-scored X
        X_t = torch.as_tensor(X_centered, dtype=torch.float64, device=dev)
        U_t, s_t, _ = torch.linalg.svd(X_t, full_matrices=False)  ## U:(n,p), s:(p,)

        ## Count numerically valid singular values.
        ## Threshold scales with index: s_j must exceed eps(U[0,0]) * j.
        eps_u1 = np.spacing(U_t[0, 0].cpu().item())
        threshold_t = eps_u1 * torch.arange(1, p_x + 1, dtype=torch.float64, device=dev)
        q = int((s_t > threshold_t).sum().item())

        ## Store everything as numpy to avoid device-mismatch issues
        ## between prefit and fit (torch upload is cheap vs SVD)
        prefit_data = {
            "X_mean_raw": X_mean_raw,       ## (p_x,)
            "X_std": X_std,                  ## (p_x,)
            "X_centered": X_centered,        ## (n, p_x)
            "U": U_t.cpu().numpy(),          ## (n, p_x)
            "s": s_t.cpu().numpy(),          ## (p_x,)
            "q": q,                          ## int
            "n": n,
            "p_x": p_x,
        }

        del X_t, U_t, s_t
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        return prefit_data

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "RidgeMML":
        """
        Fit ridge regression with MML lambda estimation.

        If ``prefit_X`` was provided at construction (or ``prefit()`` was
        called), the cached X preprocessing and SVD are reused and the
        ``X`` argument is ignored.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Design matrix (no intercept column). Shape: ``(n, p_x)``.
                Ignored when prefit data is available.
            y (Union[np.ndarray, torch.Tensor]):
                Outcome matrix. Shape: ``(n, p_y)`` or ``(n,)``.

        Returns:
            (RidgeMML):
                Fitted estimator (``self``).

        Raises:
            NotImplementedError:
                If ``fit_intercept`` is ``False``.
            ValueError:
                If shapes are incompatible or inputs are not 2-D.
        """
        if not self.fit_intercept:
            raise NotImplementedError(
                "RidgeMML requires fit_intercept=True. The MML algorithm "
                "intrinsically mean-centers the data."
            )

        ## Input validation and conversion to numpy float64
        input_is_torch = isinstance(X, torch.Tensor)
        input_device = X.device if input_is_torch else None

        if input_is_torch:
            y_np = y.detach().cpu().numpy().astype(np.float64)
        else:
            y_np = np.asarray(y, dtype=np.float64)

        if y_np.ndim == 1:
            y_np = y_np[:, None]  ## (n,) -> (n, 1)
        if y_np.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D, got ndim={y_np.ndim}")

        dev = _select_device(self.device)

        ## Use prefit data if available, otherwise preprocess X now
        if self._prefit_data is not None:
            pf = self._prefit_data
            X_mean_raw = pf["X_mean_raw"]
            X_std = pf["X_std"]
            X_centered = pf["X_centered"]
            U_np = pf["U"]
            s_np = pf["s"]
            q = pf["q"]
            n = pf["n"]
            p_x = pf["p_x"]
        else:
            if input_is_torch:
                X_np = X.detach().cpu().numpy().astype(np.float64)
            else:
                X_np = np.asarray(X, dtype=np.float64)

            if X_np.ndim != 2:
                raise ValueError(f"X must be 2-D, got ndim={X_np.ndim}")

            n, p_x = X_np.shape

            ## Z-score and mean-center X (ddof=1 for sample std).
            ## Constant columns get std=1 to avoid division by zero.
            X_mean_raw = X_np.mean(axis=0)  ## (p_x,) -- saved for intercept derivation
            X_std = np.std(X_np, axis=0, ddof=1)  ## (p_x,)
            X_std[X_std == 0] = 1.0
            X_zscored = X_np / X_std[None, :]  ## (n, p_x)
            X_centered = X_zscored - X_zscored.mean(axis=0)[None, :]  ## (n, p_x)

            ## SVD will be computed below if needed; set to None as sentinel
            U_np = None
            s_np = None
            q = None

        if y_np.shape[0] != n:
            raise ValueError(
                f"Shape mismatch: X has {n} rows, y has {y_np.shape[0]} rows"
            )

        p_y = y_np.shape[1]
        self.n_features_in_ = p_x

        ## Mean-center Y
        Y_mean = y_np.mean(axis=0)  ## (p_y,)
        Y_centered = y_np - Y_mean[None, :]  ## (n, p_y)

        ## Resolve lambdas: pre-supplied, single-regressor shortcut, or MML
        compute_lambdas = True
        lambdas_precomputed = None

        if self.lambdas is not None:
            lambdas_precomputed = np.asarray(self.lambdas, dtype=np.float64)
            if not np.isnan(lambdas_precomputed[0]):
                compute_lambdas = False
                if len(lambdas_precomputed) != p_y:
                    raise ValueError(
                        f"lambdas has length {len(lambdas_precomputed)}, "
                        f"expected {p_y} (number of y columns)"
                    )

        if p_x == 1 and compute_lambdas:
            compute_lambdas = False
            lambdas_precomputed = np.ones(p_y)

        ## Lambda optimization via MML
        convergence_failures = np.zeros(p_y, dtype=bool)

        if compute_lambdas:
            Y_t = torch.as_tensor(Y_centered, dtype=torch.float64, device=dev)

            ## Use cached SVD if available, otherwise compute now
            if U_np is not None:
                U_t = torch.as_tensor(U_np, dtype=torch.float64, device=dev)
                s_t = torch.as_tensor(s_np, dtype=torch.float64, device=dev)
            else:
                X_t = torch.as_tensor(X_centered, dtype=torch.float64, device=dev)
                U_t, s_t, _ = torch.linalg.svd(X_t, full_matrices=False)  ## U:(n,p), s:(p,)
                del X_t

                ## Count numerically valid singular values.
                ## Threshold scales with index: s_j must exceed eps(U[0,0]) * j.
                eps_u1 = np.spacing(U_t[0, 0].cpu().item())
                threshold_t = eps_u1 * torch.arange(1, p_x + 1, dtype=torch.float64, device=dev)
                q = int((s_t > threshold_t).sum().item())

            d2_t = s_t ** 2  ## (p,)
            ## alpha = diag(S) @ U^T @ Y (Karabatsos 2017, Eq. 15)
            alpha2_t = (s_t[:, None] * (U_t.T @ Y_t)) ** 2  ## (p, p_y)
            Y_var_t = torch.sum(Y_t ** 2, dim=0)  ## (p_y,)

            ## Coarse sweep for bracket bounds, then golden-section refinement
            bounds_t, hit_cap_t = _vectorized_nll_sweep(
                q=q, d2=d2_t[:q], alpha2=alpha2_t[:q, :], n=n, Y_var=Y_var_t,
            )
            lambdas_opt_t, gs_failed_t = _vectorized_golden_section(
                d2=d2_t[:q], alpha2=alpha2_t[:q, :], n=n, Y_var=Y_var_t,
                q=q, bounds=bounds_t,
            )

            lambdas_final = lambdas_opt_t.cpu().numpy()  ## (p_y,)
            convergence_failures = (hit_cap_t | gs_failed_t).cpu().numpy()

            del Y_t, U_t, s_t, d2_t, alpha2_t, Y_var_t
            del bounds_t, hit_cap_t, lambdas_opt_t, gs_failed_t
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        else:
            lambdas_final = lambdas_precomputed

        ## Beta solve: (X^T X + lambda_i * I) @ beta_i = X^T Y_i
        X_t = torch.as_tensor(X_centered, dtype=torch.float64, device=dev)
        Y_t = torch.as_tensor(Y_centered, dtype=torch.float64, device=dev)
        lambdas_t = torch.as_tensor(lambdas_final, dtype=torch.float64, device=dev)

        betas_t = _batched_solve(
            XTX=X_t.T @ X_t,
            ep=torch.eye(p_x, dtype=torch.float64, device=dev),
            lambdas=lambdas_t,
            XTY=X_t.T @ Y_t,
            batch_size=self.batch_size_solve,
        )  ## (p_x, p_y)

        ## Undo z-scoring: betas were fit on X/X_std, so divide by X_std to
        ## get coefficients that operate on raw (unscaled) X.
        X_std_t = torch.as_tensor(X_std, dtype=torch.float64, device=dev)  ## (p_x,)
        betas_t = betas_t / X_std_t[:, None]  ## (p_x, p_y)

        ## Intercept: predict(X_raw) = X_raw @ coef_ + intercept_
        ##   = X_raw @ coef_ + Y_mean - X_mean_raw @ coef_
        ##   = (X_raw - X_mean_raw) @ coef_ + Y_mean
        ## which reproduces the centered regression.
        X_mean_raw_t = torch.as_tensor(X_mean_raw, dtype=torch.float64, device=dev)
        Y_mean_t = torch.as_tensor(Y_mean, dtype=torch.float64, device=dev)
        intercept_t = Y_mean_t - X_mean_raw_t @ betas_t  ## (p_y,)

        ## Store results, matching input type
        coef_np = betas_t.cpu().numpy()
        intercept_np = intercept_t.cpu().numpy()

        del X_t, Y_t, lambdas_t, betas_t, X_std_t, X_mean_raw_t, Y_mean_t, intercept_t
        if dev.type == "cuda":
            torch.cuda.empty_cache()

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
        def _to_np(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        self.coef_ = _to_np(self.coef_)
        self.intercept_ = _to_np(self.intercept_)
        if hasattr(self, "lambdas_"):
            self.lambdas_ = _to_np(self.lambdas_)
        if hasattr(self, "convergence_failures_"):
            self.convergence_failures_ = _to_np(self.convergence_failures_)
        return self
