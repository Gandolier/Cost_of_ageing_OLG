import numpy as np
from scipy.optimize import brentq, minimize_scalar

from Individual_level.Consumption import get_consumption_path
from Individual_level.Labour import get_labour_supply
from Individual_level.Savings import get_savings_path
from main import reimport


class Household:
    r"""
    Represents the representative household's lifecycle problem in the Steady State.
    Coordinates Consumption, Labour, and Savings decisions.

    The solver penalizes solutions with negative savings (borrowing) to guide
    toward economically feasible consumption paths.
    """
    reimport()

    # Penalty multiplier for negative savings in the objective function
    BORROWING_PENALTY_WEIGHT = 100.0

    def __init__(self, p_params: dict, rho: np.array):
        r"""
        :param p_params: Dictionary of household parameters
                         (beta, sigma, rho, g_y, ltilde, b_ellip, upsilon, taxes...)
        """
        self.params = p_params
        self.S = self.params['S']
        self.E = self.params['E']
        self.rho = rho

    def solve_decisions(
            self,
            c1_guess: float,
            w: float,
            r: float,
            X_vec: np.array,
            BQ_val: float,
            clamp_savings: bool = False,
            debug_savings: bool = False,
            debug_prefix: str = ""
    ):
        r"""
        Given an initial consumption guess c1, calculate the full path of variables.

        :param clamp_savings: If True, clamps savings to be non-negative in the path.
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"

        # In SS, r is constant across ages/time, but we support vector if needed
        if isinstance(r, float):
            r_vec = np.full(self.S, r)
        else:
            r_vec = r

        assert len(r_vec) == self.S, f"r_vec length {len(r_vec)} must match S {self.S}"

        # 1. Consumption Path (Euler Equation)
        c_vec = get_consumption_path(c1_guess, r_vec, self.rho, self.params)

        # 2. Labour Supply (Intratemporal FOC)
        n_vec = get_labour_supply(w, c_vec, self.params)

        # 3. Savings Path (Budget Constraint)
        b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_val, self.params,
                                 clamp_to_zero=clamp_savings, debug=debug_savings, debug_prefix=debug_prefix)

        return c_vec, n_vec, b_vec

    def get_last_period_savings(self, c1_guess: float, w: float, r: float,
                                 X_vec: np.array, BQ_val: float) -> float:
        r"""
        Error function for the rootfinder (Shooting Method).
        Target: Savings at end of life (b_{S+1}) should be 0.

        This function includes a penalty for negative savings (borrowing).
        The penalty is negative (subtractive) so that when there's borrowing,
        the error becomes more negative, pushing Brent's toward lower c1
        (more conservative consumption → more savings → less borrowing).

        The penalty preserves monotonicity: higher c1 → lower (more negative) return.
        """
        try:
            _, _, b_vec = self.solve_decisions(c1_guess, w, r, X_vec, BQ_val, clamp_savings=False)

            # Terminal condition: b_{S+1}
            terminal = b_vec[self.S]

            # Borrowing penalty: sum of negative savings (excluding terminal)
            # np.minimum gives 0 for positive, negative value for negative savings
            # Summing these gives a negative number when there's borrowing
            negative_savings = np.minimum(b_vec[self.E:-1], 0.0)
            borrowing_penalty = np.sum(negative_savings) * self.BORROWING_PENALTY_WEIGHT

            # Return terminal + penalty (penalty is negative when there's borrowing)
            # This makes the error more negative when there's borrowing,
            # guiding the solver toward lower c1
            return terminal + borrowing_penalty

        except (ValueError, FloatingPointError):
            # Return high penalty if path is invalid (e.g. complex numbers)
            return 1e10

    def get_squared_error(self, c1_guess: float, w: float, r: float,
                          X_vec: np.array, BQ_val: float) -> float:
        r"""
        Squared error objective for minimization-based fallback.
        Minimizes (terminal + penalty)^2.
        """
        error = self.get_last_period_savings(c1_guess, w, r, X_vec, BQ_val)
        return error ** 2

    def solve_steady_state(
            self,
            w: float,
            r: float,
            X_vec: np.array,
            BQ_val: float,
            c_init_guess_range: tuple = (1e-5, 50.0),
            debug_savings: bool = False,
            debug_prefix: str = ""
    ):
        r"""
        Inner loop rootfinder algorithm that finds the optimal c_{E+1} (c1)
        such that the lifetime budget constraint holds (b_{last} = 0).

        Strategy:
        1. Evaluate error at bracket endpoints
        2. If signs differ, use Brent's method (guaranteed convergence)
        3. If signs are the same, use bounded scalar minimization as fallback
        4. The embedded borrowing penalty guides both methods toward feasible solutions

        Ref: algo:steady_state_solution
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"
        assert c_init_guess_range[0] < c_init_guess_range[1], "c_init_guess_range must be (low, high) with low < high"

        c_low, c_high = c_init_guess_range

        # Evaluate error at bracket endpoints
        f_low = self.get_last_period_savings(c_low, w, r, X_vec, BQ_val)
        f_high = self.get_last_period_savings(c_high, w, r, X_vec, BQ_val)

        c_optimal = None

        # Case 1: Opposite signs - Brent's method will find the root
        if f_low * f_high < 0:
            try:
                c_optimal = brentq(
                    self.get_last_period_savings,
                    c_low, c_high,
                    args=(w, r, X_vec, BQ_val),
                    xtol=1e-5
                )
            except ValueError:
                pass  # Fall through to minimization

        # Case 2: Same signs - use minimization to find best c1
        # This handles cases where:
        # - Both positive: c1 range might need to be higher
        # - Both negative: c1 range might need to be lower
        # - Numerical issues prevent sign change
        if c_optimal is None:
            result = minimize_scalar(
                self.get_squared_error,
                bounds=(c_low, c_high),
                method='bounded',
                args=(w, r, X_vec, BQ_val),
                options={'xatol': 1e-6}
            )
            c_optimal = result.x

            # Check solution quality
            final_error = self.get_last_period_savings(c_optimal, w, r, X_vec, BQ_val)
            if abs(final_error) > 1e-2:
                import warnings
                warnings.warn(
                    f"Household solver: Could not find exact root. "
                    f"Error = {final_error:.4f}. "
                    f"Consider adjusting c_init_guess_range or checking parameters."
                )

        # Return the decision paths
        # Important: only print savings debug for the final optimal path (avoid spamming during Brent/minimize evals)
        return self.solve_decisions(
            c_optimal, w, r, X_vec, BQ_val, clamp_savings=False,
            debug_savings=debug_savings, debug_prefix=debug_prefix
        )