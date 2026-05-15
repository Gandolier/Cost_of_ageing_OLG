I've re-read your codebase. Below is the exact specification for the coding agent. I've structured it so each section maps to a single file change with concrete signatures and the math each implementation must satisfy. Read the verification plan at the end carefully — the habit-Euler change is invasive enough that without strict h=0 regression testing, you won't know if the new code is correct or just appears correct.

## High-Level Architectural Change

The household problem currently uses forward shooting: given a single guess `c1_guess` for $\hat c_{E+1}$, iterate the Euler forward to recover all later consumptions, then check the terminal $b_{S+1}=0$ via Brent's method on the scalar `c1_guess`. With internal habits, the Euler becomes three-period coupled and forward iteration is no longer valid. We replace the entire scalar shooting layer with a global vector root-finder over $\{\hat c_s\}_{s=E}^{S-1}$ (length $S-E = 80$), solving $S-E-1$ stationary Euler residuals plus one terminal $b_{S+1}=0$ residual simultaneously via `scipy.optimize.root`.

## Parameter Addition

Add `h` (the habit parameter, a float in $[0, 1)$) to the `params` dict throughout the model. The agent's `params` must carry it; the SS solver must validate it.

In `Steady_state_equilibrium/SS_Solver.py`, the `_validate_init_inputs` method's `required_params` tuple must be extended to include `'h'`. Add the assertion `assert 0 <= params['h'] < 1, f"h must be in [0,1), got {params['h']}"` next to the other parameter-range checks.

In all callers that build a `params` dict (notebooks and `Calibration/Labour_func.py`'s `calibrate_chi` function), `h` must be present. Default value for backward-compatibility regression testing: `h = 0.0`. Production calibration value: probably `h = 0.5` initially, walked up from there.

## New File: `Individual_level/HabitUtility.py`

Create this file with two pure functions. They are the core mathematical machinery and should be unit-testable in isolation.

```python
import numpy as np


def compute_delta_c(c_vec: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the stationary habit-adjusted consumption:
        Delta_c[s] = c_vec[s] - h * exp(-g_y) * c_vec[s-1]
    with the boundary c_vec[E-1] = 0 (no initial habit), which the existing
    zero-initialization of c_vec[:E] already provides.

    Returns array of shape (S,) with zeros for s < E.
    """
    h = params['h']
    g_y = params['g_y']
    E = params['E']
    S = params['S']

    delta_c = np.zeros(S)
    # Vectorised: delta_c[E] uses c_vec[E-1] = 0, automatic
    delta_c[E:] = c_vec[E:] - h * np.exp(-g_y) * c_vec[E-1:S-1]
    return delta_c


def compute_M_from_c(c_vec: np.ndarray, rho: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the effective marginal utility of consumption:
        M[s] = Delta_c[s]^(-sigma) - h * beta * (1-rho[s]) * exp(-sigma*g_y) * Delta_c[s+1]^(-sigma)
    for s in [E, S-2], with boundary M[S-1] = Delta_c[S-1]^(-sigma) (no future).

    Returns array of shape (S,) with zeros for s < E.
    Raises ValueError if Delta_c[s] <= 0 for any s >= E (habit too strong / bad c profile).
    """
    h = params['h']
    beta = params['beta']
    sigma = params['sigma']
    g_y = params['g_y']
    E = params['E']
    S = params['S']

    delta_c = compute_delta_c(c_vec, params)

    if np.any(delta_c[E:] <= 0):
        bad = np.where(delta_c[E:] <= 0)[0] + E
        raise ValueError(
            f"Habit-adjusted consumption non-positive at ages {bad.tolist()}; "
            f"h={h} is too large for this consumption profile."
        )

    F = np.zeros(S)
    F[E:] = delta_c[E:] ** (-sigma)

    M = np.zeros(S)
    # For s in [E, S-2]: include habit-correction term
    M[E:S-1] = F[E:S-1] - h * beta * (1 - rho[E:S-1]) * np.exp(-sigma * g_y) * F[E+1:S]
    # Boundary: no future contribution
    M[S-1] = F[S-1]
    return M
```

Mathematical sanity: at $h=0$, `delta_c` reduces to `c_vec` and `M` reduces to `c_vec ** (-sigma)`, recovering the existing no-habit marginal utility.

## Modify `Individual_level/Labour.py`

The labor FOC changes from depending on $\hat c_s^{-\sigma}$ (i.e., $1/\hat c_s^\sigma$) to depending on $M_s$. Replace the entire `get_labour_supply` function with this signature and body:

```python
import numpy as np


def get_labour_supply(w: float, M_vec: np.ndarray, p_params: dict) -> np.ndarray:
    r"""
    Calculates stationary labour supply n_s given the effective marginal utility M_vec.

    New FOC (with habits + chi_s):
        n_s = l_tilde * [1 + (chi_s * b / (l_tilde * w * (1 - tau_l) * M_s))^(upsilon/(upsilon-1))]^(-1/upsilon)

    At h=0, M_s = c_s^(-sigma), and the formula reduces to the original.
    """
    S = p_params['S']
    E = p_params['E']
    l_tilde = p_params['ltilde']
    b_ellip = p_params['b_ellip']
    upsilon = p_params['upsilon']
    tau_l = p_params['tau_l']
    chi_s = p_params['chi_s'][E:]

    net_wage = w * (1 - tau_l)
    assert net_wage > 0, "Net wage must be strictly positive"
    assert np.all(M_vec[E:] > 0), "M_vec must be strictly positive over [E, S)"

    n_vec = np.zeros(S)

    # Inside parenthesis of the FOC: chi_s * b / (l_tilde * net_wage * M_s)
    numer = chi_s * b_ellip
    denom = l_tilde * net_wage * M_vec[E:]
    ratio = np.maximum(numer / denom, 1e-10)

    exponent = upsilon / (upsilon - 1)
    n_vec[E:] = l_tilde * (1 + ratio ** exponent) ** (-1 / upsilon)

    return n_vec
```

The signature change is breaking: callers must now pass `M_vec` instead of `c_vec`. The only caller is `Household.solve_decisions`, which we're rewriting anyway.

## `Individual_level/Savings.py`

No changes. The budget constraint does not depend on utility. Verify by inspection that nothing in `get_savings_path` references `sigma`, `beta`, or `h`.

## `Individual_level/Consumption.py`

Keep `get_consumption_path` unchanged. Its purpose changes: it is no longer the primary consumption solver but a helper for generating initial guesses. Add a docstring note at the top:

```python
"""
get_consumption_path: NO-HABIT forward iteration of the stationary Euler.
After the habit-formation refactor, this function is used ONLY to generate
initial guesses for the global root-finder in Households.solve_steady_state.
The habit-formation Euler is solved globally via scipy.optimize.root using
compute_M_from_c() from HabitUtility.py.
"""
```

## Major Rewrite: `Individual_level/Households.py`

This is the largest change. The file currently has `build_bq_receipt_vec` (keep unchanged), `Household.solve_decisions` (rewrite signature), `Household.get_last_period_savings` (delete), `Household.get_squared_error` (delete), and `Household.solve_steady_state` (rewrite to use vector root-finder).

The new file should look like:

```python
import numpy as np
import warnings
from scipy.optimize import root

from Individual_level.Consumption import get_consumption_path
from Individual_level.Labour import get_labour_supply
from Individual_level.Savings import get_savings_path
from Individual_level.HabitUtility import compute_M_from_c
from main import reimport


def build_bq_receipt_vec(BQ_val, omega, E, R):
    """Distribute aggregate bequests across working-age cohorts only.
    UNCHANGED from existing implementation."""
    N_working = np.sum(omega[E:R])
    bq_per_recipient = BQ_val / N_working if N_working > 0 else 0.0
    bq_receipt = np.zeros_like(omega)
    bq_receipt[E:R] = bq_per_recipient
    return bq_receipt


class Household:
    reimport()

    def __init__(self, p_params: dict, rho: np.array):
        self.params = p_params
        self.S = self.params['S']
        self.E = self.params['E']
        self.rho = rho
        # Cache last-converged c_vec for warm-starting subsequent SS iterations
        self._c_vec_cache = None

    def solve_decisions(self, c_vec_active, w, r, X_vec, BQ_vec):
        """
        Given a FULL consumption profile (c_vec_active of length S-E covering ages E to S-1),
        compute consistent labour and savings.

        Returns (c_vec, n_vec, b_vec) where c_vec is length S (zeros below E),
        n_vec is length S, b_vec is length S+1.
        """
        c_vec = np.zeros(self.S)
        c_vec[self.E:] = c_vec_active

        # Compute effective marginal utility from consumption
        M_vec = compute_M_from_c(c_vec, self.rho, self.params)

        # Labour from new FOC (uses M_vec, not c_vec directly)
        n_vec = get_labour_supply(w, M_vec, self.params)

        # Savings from budget (unchanged)
        r_vec = np.full(self.S, r) if np.isscalar(r) else r
        b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_vec, self.params)

        return c_vec, n_vec, b_vec

    def euler_residuals(self, c_vec_active, w, r, X_vec, BQ_vec):
        """
        Residual vector of length S-E for the global root-finder.

        First S-E-1 entries: stationary habit Euler residuals
            M[s] - beta*(1-rho[s])*(1+r_net)*exp(-sigma*g_y)*M[s+1]    for s in [E, S-2]

        Last entry: terminal savings condition
            b_vec[S] = 0

        Args:
            c_vec_active: array of shape (S-E,) representing c_vec[E:S].
        """
        sigma = self.params['sigma']
        beta = self.params['beta']
        g_y = self.params['g_y']
        tau_k = self.params['tau_k']
        E, S = self.E, self.S

        try:
            c_vec, n_vec, b_vec = self.solve_decisions(c_vec_active, w, r, X_vec, BQ_vec)
            M_vec = compute_M_from_c(c_vec, self.rho, self.params)
        except (ValueError, FloatingPointError) as e:
            # Habit-adjusted consumption went negative or other numerical pathology.
            # Return a large residual to push the solver away from this region.
            return np.full(S - E, 1e6)

        r_net = r * (1 - tau_k)

        # Euler residuals for s in [E, S-2]
        euler_errs = M_vec[E:S-1] - beta * (1 - self.rho[E:S-1]) * (1 + r_net) \
                                  * np.exp(-sigma * g_y) * M_vec[E+1:S]

        # Terminal condition: savings at end of life must be zero
        terminal_err = b_vec[S]

        return np.concatenate([euler_errs, [terminal_err]])

    def _initial_guess(self, w, r, X_vec, BQ_vec):
        """Build an initial guess for c_vec[E:S] from the no-habit forward iterator."""
        r_vec = np.full(self.S, r)
        # Use a moderate starting consumption; the root-finder will refine.
        # We pick the midpoint of a sensible range based on labour income at age E.
        # Alternative: any positive flat profile works; the solver is robust.
        c_start_guess = 0.5  # tune if needed; this is just a starting point
        c_vec_full = get_consumption_path(c_start_guess, r_vec, self.rho, self.params)
        return c_vec_full[self.E:]

    def solve_steady_state(self, w, r, X_vec, BQ_val, omega,
                           c_init=None, debug_savings=False, debug_prefix=""):
        """
        Solve the habit-formation household problem by globally root-finding
        the consumption profile c_vec[E:S].

        Args:
            c_init: optional warm-start guess for c_vec[E:S]. If None, uses
                    either the cached previous solution (self._c_vec_cache)
                    or the no-habit forward iteration.

        Returns: (c_vec, n_vec, b_vec) with shapes (S,), (S,), (S+1,).
        """
        E, S = self.E, self.S
        BQ_vec = build_bq_receipt_vec(BQ_val, omega, E, self.params['R'])

        # Choose initial guess: explicit > cached > no-habit forward iteration
        if c_init is not None:
            x0 = np.asarray(c_init).copy()
        elif self._c_vec_cache is not None:
            x0 = self._c_vec_cache.copy()
        else:
            x0 = self._initial_guess(w, r, X_vec, BQ_vec)

        # Robust root-finder. 'hybr' (Powell hybrid) is the default and usually works.
        sol = root(
            self.euler_residuals,
            x0,
            args=(w, r, X_vec, BQ_vec),
            method='hybr',
            options={'xtol': 1e-8, 'maxfev': 2000},
        )

        if not sol.success:
            warnings.warn(
                f"Household root-finder failed: {sol.message}. "
                f"Residual norm: {np.linalg.norm(sol.fun):.4e}. "
                f"Falling back to LM method."
            )
            sol = root(
                self.euler_residuals,
                x0,
                args=(w, r, X_vec, BQ_vec),
                method='lm',
                options={'xtol': 1e-8, 'maxiter': 2000},
            )

        if not sol.success:
            warnings.warn(
                f"Household root-finder failed with both methods. "
                f"Final residual norm: {np.linalg.norm(sol.fun):.4e}."
            )

        # Cache for next call
        self._c_vec_cache = sol.x.copy()

        # Final reconstruction for return
        return self.solve_decisions(sol.x, w, r, X_vec, BQ_vec)
```

Key points the agent must respect:

1. The borrowing penalty mechanism in the old `get_last_period_savings` is GONE. The global root-finder doesn't need it. If post-conversion the new solver lands on negative-savings solutions, that's a different problem to solve, not a hidden constraint to add back.

2. The `solve_decisions` signature changed: it now takes a vector `c_vec_active` of length $S-E$, not a scalar `c1_guess`. Every existing caller of `solve_decisions` must be updated.

3. The cache `_c_vec_cache` is what gives convergence speed in the outer SS loop. Without it, every iteration starts cold and the inner solver is 10x slower.

## `Steady_state_equilibrium/SS_Solver.py` Changes

Only three changes here:

First, in `_validate_init_inputs`, add `'h'` to the `required_params` tuple and add the range assertion.

Second, the call to `self.household.solve_steady_state(...)` in the outer loop no longer needs `c_init_guess_range`. Remove that argument. The new function uses the household's internal cache for warm-starting.

Third, `check_euler_errors` is broken under habits because it uses the no-habit Euler formula. Replace its body with:

```python
def check_euler_errors(self, ss_dict: dict):
    """Validates the Habit-Formation Euler Equation."""
    from Individual_level.HabitUtility import compute_M_from_c

    c_vec = ss_dict['c_vec']
    r = ss_dict['r']
    sigma = self.params['sigma']
    beta = self.params['beta']
    g_y = self.params['g_y']
    tau_k = self.gov.tau_k
    E, S = self.params['E'], self.params['S']

    M_vec = compute_M_from_c(c_vec, self.rho, self.params)
    r_net = r * (1 - tau_k)

    lhs = M_vec[E:S-1]
    rhs = beta * (1 - self.rho[E:S-1]) * (1 + r_net) * np.exp(-sigma * g_y) * M_vec[E+1:S]
    euler_errs = lhs - rhs

    max_err = np.max(np.abs(euler_errs))
    print(f"Habit Euler Equation: Max Abs Error = {max_err:.6e}")
    return euler_errs
```

## `Calibration/Labour_func.py` Changes

The `update_chi_from_foc` function inverts the labor FOC for $\chi_s$. With habits, the FOC depends on $M_s$ instead of $\hat c_s^{-\sigma}$. The change:

```python
def update_chi_from_foc(c_vec, w, tau_l, n_target, rho, params):
    """
    Closed-form inverse of the habit-modified labor FOC.

    chi_s = [l_tilde * (1 - tau_l) * w * M_s / b] * (eta_s^(-upsilon) - 1)^((upsilon-1)/upsilon)

    Note new argument 'rho' (needed to compute M_s). At h=0, M_s reduces to c_s^(-sigma)
    and the formula reduces to the original.
    """
    from Individual_level.HabitUtility import compute_M_from_c

    upsilon = params['upsilon']
    b = params['b_ellip']
    l_tilde = params['ltilde']
    E = params['E']
    S = params['S']

    chi_s = np.ones(S)
    eta = np.clip(n_target[E:] / l_tilde, 1e-6, 1 - 1e-6)
    leisure_term = (eta ** (-upsilon) - 1) ** ((upsilon - 1) / upsilon)

    M_vec = compute_M_from_c(c_vec, rho, params)

    chi_s[E:] = (l_tilde * (1 - tau_l) * w * leisure_term * M_vec[E:]) / b
    return chi_s
```

The signature gains a `rho` argument. The caller `calibrate_chi` must pass `vectors['rho']` to it:

```python
chi_s_new = update_chi_from_foc(
    c_vec=result['c_vec'],
    w=result['w'],
    tau_l=result['tau_l'],
    n_target=n_target,
    rho=vectors['rho'],   # <-- new
    params=params,
)
```

## Implementation Order and Verification

This is the critical part. The agent must implement these in the order below, running the verification after each step, because a broken step propagates silently and is painful to debug.

**Stage 1 — Plumb the parameter.** Add `h` to params dict with default `0.0`. Add validation in SS_Solver. Verify the existing model runs unchanged with `h = 0.0` set explicitly. *Pass criterion: the equilibrium $r$, $\hat K$, $\hat L$, and entire $c_vec$ are bit-identical to the pre-change run.*

**Stage 2 — Create `HabitUtility.py`.** Write `compute_delta_c` and `compute_M_from_c`. Write a unit test that at `h = 0`, `compute_M_from_c(c_vec, rho, params)` returns exactly `c_vec ** (-sigma)` for $s \in [E, S)$ and zero elsewhere. *Pass criterion: unit test passes, no callers modified yet.*

**Stage 3 — Refactor Labour.py.** Change `get_labour_supply` to take `M_vec`. Update its sole caller (`Household.solve_decisions`) to compute `M_vec` from `c_vec` first, then pass it in. Run the existing SS solver at `h = 0`. *Pass criterion: identical equilibrium to Stage 1, since at `h=0`, `M_vec = c_vec ** (-sigma)` and the labor formula is mathematically equivalent.*

**Stage 4 — Rewrite Households.py with global root-finder.** Replace forward shooting with `scipy.optimize.root` over the c_vec[E:S] vector. Add Euler residual function. Add warm-start cache. Run the existing SS solver at `h = 0`. *Pass criterion: equilibrium $r$, $\hat K$, $\hat L$, and $c_vec$ match Stages 1-3 to within $10^{-5}$ relative error. This is the most important verification step — if the new solver doesn't match the old solver at h=0, there is a bug in the new code, and you must find it before proceeding.*

**Stage 5 — Update SS_Solver.** Adjust the outer-loop call to use the new `solve_steady_state` signature (drop `c_init_guess_range`). Update `check_euler_errors`. Run at `h = 0`. *Pass criterion: same as Stage 4.*

**Stage 6 — Update chi calibration.** Modify `update_chi_from_foc` to accept `rho`. Run `calibrate_chi` at `h = 0` from start to converged $\chi_s$. *Pass criterion: the calibrated $\chi_s$ profile is bit-identical (or within $10^{-5}$) to the pre-change calibration. If not, there is a bug in the new formula.*

**Stage 7 — Turn habits on.** Set `h = 0.1` and re-solve the SS. Inspect the consumption profile. *Pass criterion: the consumption profile is slightly flatter than at `h = 0`, the equilibrium $r$ has moved slightly downward (more savings due to habit smoothing), and the root-finder converges in fewer than 50 iterations of `hybr`. If the root-finder fails to converge, try `h = 0.05` first, then walk up.*

**Stage 8 — Walk to target.** Increment `h` in steps of `0.1` up to your target value (say `0.5`), re-solving each time, using the previous solution as the initial guess (the cache handles this automatically). Re-run the $\chi_s$ calibration. *Pass criterion: convergence at each step, monotonically flatter consumption profile.*

## Things the Agent Must NOT Change

To prevent scope creep: do not modify `Aggregates.py`, `Production.py`, `PublicSector.py`, or `Demography.py`. These do not depend on utility specification. Do not modify the bequest distribution mechanism (`build_bq_receipt_vec`); that was the previous round's change and is correct. Do not change the relaxation factor `xi` or other outer-loop tuning unless the habit version fails to converge.

Do not delete `get_consumption_path` in `Consumption.py`. It remains useful as an initial-guess generator. Just leave it, and add the docstring note.

Do not optimise the root-finder's performance until you have correctness. The `hybr` method with `xtol=1e-8` is the default; don't switch to faster methods that may be less robust until Stage 7 is passing reliably.

## Required Sanity Output at Each Stage

After every stage, the agent must report the following three numbers from a clean SS solve with `h = 0`:
- The equilibrium interest rate $r$ (to 6 decimal places).
- The equilibrium aggregate capital $\hat K$ (to 4 decimal places).
- The L1 norm of `c_vec` (i.e., `np.sum(np.abs(c_vec))`).

Stages 1 through 6 must all produce identical values for these three numbers (to numerical precision). If they don't, halt and debug — the implementation has a bug, and any habit-related behavior observed thereafter is meaningless.

Once `h > 0` (Stage 7+), these numbers will change, and the comparison switches to: did the root-finder converge, and is the consumption profile monotonically flatter than the no-habit baseline.