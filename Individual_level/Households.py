import numpy as np
import warnings
from scipy.optimize import brentq, root

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
    """
    reimport()

    # Same value as the pre-habit shooter used; chosen large enough that
    # the root-finder is unambiguously pushed away from borrowing paths.
    BORROWING_PENALTY_WEIGHT = 100.0

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

        # Effective marginal utility (habit-aware; at h=0 reduces to c^(-sigma))
        M_vec = compute_M_from_c(c_vec, self.rho, self.params)

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

    def _enforce_delta_c_positive(self, c_active: np.ndarray) -> np.ndarray:
        r"""
        Enforce $\widehat{\Delta c}_s > 0$ for all $s \in [E, S)$ by
        clamping the consumption growth rate from below.

        At $h > 0$, feasibility requires $c_s / c_{s-1} > h e^{-g_y}$.
        The no-habit forward iterator can violate this at old ages where
        high mortality makes the Euler growth factor very small.  This
        helper post-processes the profile so that the root-finder starts
        in the feasible region (§10.1).
        """
        h = self.params.get('h', 0.0)
        if h <= 0:
            return c_active
        g_y = self.params['g_y']
        min_ratio = h * np.exp(-g_y) + 1e-4
        out = c_active.copy()
        for k in range(1, len(out)):
            out[k] = max(out[k], min_ratio * out[k - 1])
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

        If the brent bracket fails (no sign change on `b_{S+1}`), fall
        back to a positive flat profile.
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
        # When the bracket does not straddle a zero (extreme prices), we
        # fall back to a moderate forward-iterated profile and rely on the
        # subsequent `scipy.optimize.root` call to refine. This keeps the
        # cold-start logic in lock-step with the pre-Stage-4 behaviour at
        # h=0 (the basis of the Stage-4 pass criterion).
        c_low, c_high = 1e-5, 50.0
        try:
            f_low = terminal_b(c_low)
            f_high = terminal_b(c_high)
            if (np.isfinite(f_low) and np.isfinite(f_high)
                    and f_low * f_high < 0):
                c1 = brentq(terminal_b, c_low, c_high, xtol=1e-6)
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

        # Last-resort fallback: positive flat profile.
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
            1. Treat $\hat c_{E}, \dots, \hat c_{S-1}$ as $S-E$ unknowns.
            2. Construct $S-E$ residuals: $S-E-1$ Euler equations in $\hat M_s$
               plus one terminal $\hat b_{S+1} = 0$ residual.
            3. Solve with `scipy.optimize.root` (Powell hybrid, fallback LM).
            4. Warm-start from `c_init` > `self._c_vec_cache` > no-habit
               forward iteration.

        Returns (c_vec, n_vec, b_vec) of shapes (S,), (S,), (S+1,).
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

        # Primary: Powell hybrid (`hybr`). Robust default per spec.
        sol = root(
            self.euler_residuals,
            x0,
            args=(w, r, X_vec, BQ_vec),
            method='hybr',
            options={'xtol': 1e-8, 'maxfev': 2000},
        )

        if not sol.success:
            warnings.warn(
                f"Household root-finder (hybr) failed: {sol.message}. "
                f"Residual norm: {np.linalg.norm(sol.fun):.4e}. "
                f"Falling back to LM."
            )
            sol = root(
                self.euler_residuals,
                x0,
                args=(w, r, X_vec, BQ_vec),
                method='lm',
                options={'xtol': 1e-8, 'maxiter': 2000},
            )

        res_norm = float(np.linalg.norm(sol.fun))
        if not sol.success or not np.isfinite(res_norm) or res_norm > 1e-2:
            warnings.warn(
                f"Household root-finder failed with both methods."
                f"Final residual norm: {res_norm:.4e}."
                f"Falling back to fresh brent-shoot start."
            )
            # Bad outer-loop guess: fall back to a fresh brent-shoot start so
            # downstream aggregates remain finite. Do NOT cache.
            x_fallback = self._initial_guess(w, r, X_vec, BQ_vec)
            for x_try in [x_fallback,
                          self._enforce_delta_c_positive(np.maximum(sol.x, 1e-10)),
                          np.full(S - E, 0.5)]:
                try:
                    return self.solve_decisions(x_try, w, r, X_vec, BQ_vec)
                except (ValueError, FloatingPointError, AssertionError):
                    continue
            raise RuntimeError(
                f"Household root-finder failed (hybr + LM) and all fallbacks "
                f"exhausted. Final residual norm: {res_norm:.4e}. "
                f"solver message: {sol.message}"
            )

        # Cache converged profile for warm-starting subsequent outer iters.
        self._c_vec_cache = sol.x.copy()

        return self.solve_decisions(sol.x, w, r, X_vec, BQ_vec)
