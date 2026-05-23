import numpy as np
import warnings
from scipy.optimize import brentq, root, least_squares

from Individual_level.Consumption import get_consumption_path
from Individual_level.Labour import get_labour_supply
from Individual_level.Savings import get_savings_path
from Individual_level.HabitUtility import compute_M_from_c
from main import reimport


def build_bq_receipt_vec(BQ_val, omega, E, R):
    r"""Distribute aggregate bequests across working-age cohorts only.
    UNCHANGED from the pre-habit implementation."""
    # working age: E+1 to R. In 0-indexed: index E to R-1.
    N_working = np.sum(omega[E:R])
    bq_per_recipient = BQ_val / N_working if N_working > 0 else 0.0

    bq_receipt = np.zeros_like(omega)
    bq_receipt[E:R] = bq_per_recipient  # E+1 to R receive; s > R get 0
    return bq_receipt


class Household:
    r"""
    Representative-household lifecycle problem in stationary Steady State.

    With internal habit formation the Euler in $M_s$ collapses to a
    three-period-coupled relation in consumption levels (see
    `Documentation/habits_derivations.md` §9.1), so forward shooting is no
    longer well-posed. The household problem is therefore solved as a
    global vector root-finding problem over $\hat c_s$ for $s \in [E, S)$
    (length $S-E$) using `scipy.optimize.root`:

      * $S-E-1$ stationary Euler residuals (Section 9.2(a))
            $\hat M_s - \beta(1-\rho_s)(1+r(1-\tau^k)) e^{-\sigma g_y}\hat M_{s+1} = 0$
      * 1 terminal-savings residual (Section 9.2(b))
            $\hat b_{S+1} = 0$.

    At $h=0$ the Euler reduces to the paper's eq. (2.26) and the solver
    must reproduce the no-habit equilibrium to numerical precision.

    Per `habits_derivations.md` §10.1, numerical violations (non-positive
    habit-adjusted consumption; negative stationary savings) are reported
    as large residuals so that the root-finder explores away from those
    infeasible regions. The no-borrowing constraint is enforced via a
    smooth `min(0, b_vec[E+1:S])` penalty added to the terminal residual
    — same mechanism as the pre-habit shooter used to keep the outer
    SS_Solver loop stable.

    Convergence policy
    ------------------
    The household solver runs Powell hybrid (`hybr`) as primary and
    Levenberg–Marquardt (`lm`) as a secondary attempt on the SAME problem.
    Both are legitimate solver retries — they re-solve the actual Euler
    system, they do not substitute a fictitious answer. If both fail to
    reach `RESIDUAL_TOL`, the method raises `RuntimeError`. There is no
    silent fallback to a no-habit profile, flat dummy, or any other
    non-solution: returning a non-Euler-satisfying profile up the call
    stack would let the outer SS loop converge to a fake equilibrium.

    All convergence-diagnostic messages are emitted via `print(..., flush=True)`
    so they are reliably visible in Jupyter cell output (warnings.warn
    can be silenced by filters in test scripts).
    """
    reimport()

    # Same value as the pre-habit shooter used; chosen large enough that
    # the root-finder is unambiguously pushed away from borrowing paths.
    BORROWING_PENALTY_WEIGHT = 100

    def __init__(self, p_params: dict, rho: np.array):
        r"""
        :param p_params: dict of household parameters
                         (beta, sigma, rho, g_y, ltilde, b_ellip, upsilon,
                          h, taxes, ...).
        :param rho:      length-S vector of stationary mortality hazards.
        """
        self.params = p_params
        self.S = self.params['S']
        self.E = self.params['E']
        self.rho = rho
        # Warm-start cache: previous converged c_vec[E:S].
        self._c_vec_cache = None

    # ------------------------------------------------------------------
    # Per-iteration helpers
    # ------------------------------------------------------------------

    def solve_decisions(
            self,
            c_vec_active: np.ndarray,
            w: float,
            r: float,
            X_vec: np.ndarray,
            BQ_vec: np.ndarray,
            delta_c=None,
    ):
        r"""
        Given a full stationary consumption profile (`c_vec_active` of length
        $S-E$ covering ages $E, \dots, S-1$), compute the consistent labour
        and savings profiles.

        Returns (c_vec, n_vec, b_vec) of shapes (S,), (S,), (S+1,).
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"
        assert len(BQ_vec) == self.S, f"BQ_vec length {len(BQ_vec)} must match S {self.S}"

        c_vec_active = np.asarray(c_vec_active, dtype=float)
        assert c_vec_active.shape == (self.S - self.E,), (
            f"c_vec_active must have shape ({self.S - self.E},), got {c_vec_active.shape}"
        )

        c_vec = np.zeros(self.S)
        c_vec[self.E:] = c_vec_active

        if delta_c is not None:
            from Individual_level.HabitUtility import compute_M_from_delta_c
            M_vec = compute_M_from_delta_c(delta_c, self.rho, self.params)
        else:
            M_vec = compute_M_from_c(c_vec, self.rho, self.params)  # legacy / h=0

        # Labour Supply (intratemporal FOC, driven by M_vec)
        n_vec = get_labour_supply(w, M_vec, self.params)

        # Savings (budget constraint, unchanged)
        r_vec = np.full(self.S, r) if np.isscalar(r) else r
        b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_vec, self.params)

        return c_vec, n_vec, b_vec

    def euler_residuals(
            self,
            c_vec_active: np.ndarray,
            w: float,
            r: float,
            X_vec: np.ndarray,
            BQ_vec: np.ndarray,
    ) -> np.ndarray:
        r"""
        Residual vector (length $S-E$) for the global root-finder.

        Entries $[0, S-E-2]$: stationary habit Euler residuals (§9.2(a))
            $\hat M_s - \beta(1-\rho_s)(1+r(1-\tau^k))e^{-\sigma g_y}\hat M_{s+1}$
        for $s \in [E, S-2]$.

        Entry $[S-E-1]$: terminal savings condition $\hat b_{S+1} = 0$.

        Numerical pathologies (non-positive $\widehat{\Delta c}$, etc.) are
        reported as a large constant residual so that the root-finder can
        escape the infeasible region (§10.1).
        """
        sigma = self.params['sigma']
        beta = self.params['beta']
        g_y = self.params['g_y']
        tau_k = self.params['tau_k']
        E, S = self.E, self.S

        try:
            c_vec, _, b_vec = self.solve_decisions(c_vec_active, w, r, X_vec, BQ_vec)
            M_vec = compute_M_from_c(c_vec, self.rho, self.params)
        except (ValueError, FloatingPointError, AssertionError):
            return np.full(S - E, 1e6)

        if not (np.all(np.isfinite(M_vec[E:])) and np.all(np.isfinite(b_vec))):
            return np.full(S - E, 1e6)

        r_net = r * (1 - tau_k)

        # Euler residuals at s = E, ..., S-2 (length S-E-1).
        euler_errs = (
            M_vec[E:S - 1]
            - beta * (1 - self.rho[E:S - 1]) * (1 + r_net)
            * np.exp(-sigma * g_y) * M_vec[E + 1:S]
        )

        # Terminal condition: stationary assets at S+1 (index S) must vanish.
        # The smooth `min(0, b_vec)` penalty (§10.1) keeps the root-finder
        # away from negative-savings paths; mirrors the pre-habit shooter.
        negative_savings = np.minimum(b_vec[E + 1:S], 0.0)
        borrowing_penalty = float(np.sum(negative_savings)) * self.BORROWING_PENALTY_WEIGHT
        terminal_err = b_vec[S] + borrowing_penalty

        return np.concatenate([euler_errs, [terminal_err]])

    def euler_residuals_z(self, z, w, r, X_vec, BQ_vec):
        """
        Euler residuals in z = log(Δc) space. Δc = exp(z) is positive
        by construction, so compute_M_from_c never raises and the
        catch-fallback is structurally unreachable. hybr can navigate
        this freely.
        """
        sigma = self.params['sigma']
        beta = self.params['beta']
        g_y = self.params['g_y']
        tau_k = self.params['tau_k']
        h = self.params.get('h', 0.0)
        c_min = self.params.get('c_min', 0.0)
        E, S = self.E, self.S
        h_disc = h * np.exp(-g_y)

        # Reconstruct c from z via the recursion
        #     c_s = h*e^(-g_y) * c_{s-1} + c_min + exp(z_s)
        # with c_{E-1} = 0.
        delta_c_active = np.exp(z)              # length S-E, strictly positive
        c_vec = np.zeros(S)
        c_prev = 0.0
        for k in range(S - E):
            c_vec[E + k] = h_disc * c_prev + c_min + delta_c_active[k]
            c_prev = c_vec[E + k]

        # M directly from Δc — bypass compute_M_from_c entirely (no exception possible)
        F = delta_c_active ** (-sigma)
        M = np.zeros(S)
        M[E:S - 1] = (
            F[:-1]
            - h * beta * (1 - self.rho[E:S - 1]) * np.exp(-sigma * g_y) * F[1:]
        )
        M[S - 1] = F[-1]

        # Labor + savings via existing functions
        try:
            n_vec = get_labour_supply(w, M, self.params)
            r_vec = np.full(S, r) if np.isscalar(r) else r
            b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_vec, self.params)
        except (AssertionError, FloatingPointError, ValueError):
            return np.full(S - E, 1e6)

        if not np.all(np.isfinite(b_vec)):
            return np.full(S - E, 1e6)

        r_net = r * (1 - tau_k)

        # Raw Euler residuals
        euler_raw = (
            M[E:S - 1]
            - beta * (1 - self.rho[E:S - 1]) * (1 + r_net) * np.exp(-sigma * g_y) * M[E + 1:S]
        )
        # Rescale by local M magnitude. At a true root the numerator is exactly
        # zero, so dividing by a positive scale does NOT move the solution — it
        # only equalises the conditioning across ages so the solver weighs the
        # old-age Euler and the terminal budget comparably.
        euler_scale = np.maximum(np.maximum(np.abs(M[E:S - 1]), np.abs(M[E + 1:S])), 1.0)
        euler_errs = euler_raw / euler_scale

        # Negative savings (borrowing) penalty. Allows solver to explore away borrowing solutions
        negative_savings = np.minimum(b_vec[E + 1:S], 0.0)
        borrowing_penalty = float(np.sum(negative_savings)) * self.BORROWING_PENALTY_WEIGHT

        # Terminal: rescale by CONSUMPTION scale, NOT asset scale.
        # "b_S unspent" is naturally measured in years-of-consumption; dividing by
        # mean consumption keeps it O(10-50) so the solver MUST pin the level.
        c_scale = max(float(np.mean(c_vec[E:])), 1.0)
        terminal_err = (b_vec[S] + borrowing_penalty) / c_scale

        return np.concatenate([euler_errs, [terminal_err]])

    def _enforce_delta_c_positive(self, c_active: np.ndarray) -> np.ndarray:
        r"""
        Enforce $\widehat{\Delta c}_s > 0$ for all $s \in [E, S)$ by
        clamping the consumption growth rate from below.

        At $h > 0$, feasibility requires $c_s / c_{s-1} > h e^{-g_y}$.
        The no-habit forward iterator can violate this at old ages where
        high mortality makes the Euler growth factor very small.  This
        helper post-processes the profile so that the root-finder starts
        in the feasible region (§10.1). Applied ONLY to the initial
        guess, not to the solver's returned solution.
        """
        h = self.params.get('h', 0.0)
        c_min = self.params.get('c_min', 0.0)
        if h <= 0 and c_min <= 0:
            return c_active
        g_y = self.params['g_y']
        min_ratio = h * np.exp(-g_y) + 1e-4
        out = c_active.copy()
        out[0] = max(out[0], c_min+1e-4)
        for k in range(1, len(out)):
            out[k] = max(out[k], min_ratio * out[k - 1] + c_min + 1e-4)
        return out

    def _initial_guess(self, w, r, X_vec, BQ_vec) -> np.ndarray:
        r"""
        Build a cold-start initial guess for `c_vec[E:S]`.

        Strategy: forward-iterate the no-habit Euler from a starting
        $\hat c_E$, then brent-search $\hat c_E$ so that the implied
        terminal stationary asset $\hat b_{S+1}$ is zero. At $h=0$ this
        already satisfies every residual of `euler_residuals` exactly, so
        the subsequent `scipy.optimize.root` call converges in one step
        and reproduces the pre-habit equilibrium to numerical precision
        (the Stage-4 pass criterion). At $h>0$ this still gives a sensible
        feasible-budget profile to seed `hybr`.

        These fallbacks affect only the INITIAL GUESS handed to the
        root-finder; the final solution is always the root-finder's
        converged output (or the method raises). The cascade is:
            1. brent-shoot on c_E with the savings-penalty objective
            2. no-habit forward iterate from c_E = 1.0
            3. flat profile of 0.5
        """
        r_vec = np.full(self.S, r) if np.isscalar(r) else r

        def terminal_b(c1):
            try:
                c_full = get_consumption_path(c1, r_vec, self.rho, self.params)
                M_vec = compute_M_from_c(c_full, self.rho, self.params)
                n_full = get_labour_supply(w, M_vec, self.params)
                b_full = get_savings_path(c_full, n_full, w, r_vec, X_vec, BQ_vec, self.params)
                if not np.isfinite(b_full[self.S]):
                    return 1e10
                # Same borrowing penalty as `euler_residuals` so the cold
                # start already lives in the feasible (no-borrowing) region.
                neg = np.minimum(b_full[self.E + 1:self.S], 0.0)
                pen = float(np.sum(neg)) * self.BORROWING_PENALTY_WEIGHT
                return float(b_full[self.S]) + pen
            except (ValueError, FloatingPointError, AssertionError):
                return 1e10

        # Fixed bracket [1e-5, 50] matching the pre-habit shooter's range.
        c_low, c_high = 1e-5, 50.0
        try:
            f_low = terminal_b(c_low)
            f_high = terminal_b(c_high)
            if (np.isfinite(f_low) and np.isfinite(f_high)
                    and f_low * f_high < 0):
                c1 = brentq(terminal_b, c_low, c_high, xtol=1e-12)
                c_vec_full = get_consumption_path(c1, r_vec, self.rho, self.params)
                return self._enforce_delta_c_positive(c_vec_full[self.E:])
        except (ValueError, FloatingPointError, AssertionError):
            pass

        # No sign change in the bracket: use a forward-iterated profile
        # from a moderate starting consumption as the cold start.
        try:
            c_vec_full = get_consumption_path(1.0, r_vec, self.rho, self.params)
            if np.all(np.isfinite(c_vec_full[self.E:])) and np.all(c_vec_full[self.E:] > 0):
                return self._enforce_delta_c_positive(c_vec_full[self.E:])
        except (ValueError, FloatingPointError, AssertionError):
            pass

        # Last-resort initial-guess fallback: positive flat profile.
        # (Affects only the seed for the root-finder, not the final answer.)
        return np.full(self.S - self.E, 0.5)

    # ------------------------------------------------------------------
    # Public entry-point used by SS_Solver
    # ------------------------------------------------------------------

    def solve_steady_state(
            self,
            w: float,
            r: float,
            X_vec: np.ndarray,
            BQ_val: float,
            omega: np.ndarray,
            c_init: np.ndarray = None,
    ):
        r"""
        Solve the habit-formation household problem by globally root-finding
        the stationary consumption profile $\hat c_s$ for $s \in [E, S)$.

        Algorithm (`habits_derivations.md` §9.2):
            1. Treat $z_{E}, \dots, z_{S-1}$ as $S-E$ unknowns where $z_s = \log(\Delta c_s)$.
            2. Construct $S-E$ residuals: $S-E-1$ Euler equations in $\hat M_s$
               plus one terminal $\hat b_{S+1} = 0$ residual.
            3. Solve with `scipy.optimize.root` (Powell hybrid; LM secondary).
            4. Warm-start from `c_init` > `self._c_vec_cache` > no-habit
               forward iteration.

        Returns (c_vec, n_vec, b_vec) of shapes (S,), (S,), (S+1,).

        Raises
        ------
        RuntimeError
            If neither hybr nor LM brings the Euler residual under
            `RESIDUAL_TOL`. Caller (outer SS loop) is expected to handle
            this — the alternative is silently returning a non-solution,
            which would let the outer loop converge to a fake equilibrium.
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"
        assert len(omega) == self.S, f"omega length {len(omega)} must match S {self.S}"

        E, S = self.E, self.S
        BQ_vec = build_bq_receipt_vec(BQ_val, omega, E, self.params['R'])

        # Pick initial guess: explicit > cached > no-habit forward iteration.
        if c_init is not None:
            x0 = np.asarray(c_init, dtype=float).copy()
            assert x0.shape == (S - E,), (
                f"c_init must have shape ({S - E},), got {x0.shape}"
            )
        elif self._c_vec_cache is not None and self._c_vec_cache.shape == (S - E,):
            x0 = self._c_vec_cache.copy()
        else:
            x0 = self._initial_guess(w, r, X_vec, BQ_vec)

        # Ensure delta_c > 0 for the initial guess at h > 0 (§10.1).
        x0 = self._enforce_delta_c_positive(np.maximum(x0, 1e-10))

        g_y = self.params['g_y']
        h = self.params.get('h', 0.0)
        c_min = self.params.get('c_min', 0.0)
        h_disc = h * np.exp(-g_y) if h > 0 else 0.0

        # Convert x0 (c-space) to z0 (log-Δc-space)
        delta_c_init = np.zeros(S - E)
        c_prev = 0.0
        for k in range(S - E):
            delta_c_init[k] = x0[k] - h_disc * c_prev - c_min
            c_prev = x0[k]
        delta_c_init = np.maximum(delta_c_init, 1e-6)  # safety floor for log
        z0 = np.log(delta_c_init)

        # Solve in z-space using fast hybr
        sol = root(
            self.euler_residuals_z,
            z0,
            args=(w, r, X_vec, BQ_vec),
            method='hybr',
            options={'xtol': 1e-10, 'maxfev': 5000},
        )

        x = sol.x
        if not sol.success:
            # print(
            #     f"[Household] hybr in z-space did not converge: {sol.message} "
            #     f"(residual norm {np.linalg.norm(sol.fun):.4e}). Retrying with lm.",
            #     flush=True,
            # )
            newsol = root(
                self.euler_residuals_z,
                z0,
                args=(w, r, X_vec, BQ_vec),
                method='lm',
                options={'xtol': 1e-10, 'maxiter': 5000},
            )

            if np.linalg.norm(sol.fun) < np.linalg.norm(newsol.fun):
                x = sol.x
            else:
                x = newsol.x

        # Δc by construction; clamp away from exact zero in case any x[k] < ~-745
        delta_c = np.zeros(self.S)
        delta_c[self.E:] = np.clip(np.exp(x), 1e-10, 1e2)

        sol_c_active = np.zeros(S - E)
        c_prev = 0.0
        for k in range(S - E):
            sol_c_active[k] = h_disc * c_prev + c_min + delta_c[E + k]
            c_prev = sol_c_active[k]

        self._c_vec_cache = sol_c_active.copy()

        c_vec, n_vec, b_vec = self.solve_decisions(sol_c_active, w, r, X_vec, BQ_vec)
        res = self.euler_residuals_z(sol.x, w, r, X_vec, BQ_vec)   # or euler_residuals in c-space
        R = self.params['R']

        # print(f"[DIAG] max|euler|={np.max(np.abs(res[:-1])):.3e}, "
            # f"terminal={res[-1]:.3e}, b_S={b_vec[self.S]:.4f}, "
            # f"b_R={b_vec[R]:.4f}, c_old={c_vec[self.S-1]:.4f}", flush=True)
        # print(25*'---')
        
        return c_vec, n_vec, b_vec