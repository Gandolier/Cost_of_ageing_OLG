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

    NOTE: This implementation enforces the no-borrowing constraint (b_s >= 0 for all s).
    When the constraint binds, consumption is reduced to available resources.
    """
    reimport()
    def __init__(self, p_params: dict, rho: np.array):
        r"""
        :param p_params: Dictionary of household parameters
                         (beta, sigma, rho, g_y, ltilde, b_ellip, upsilon, taxes...)
        """
        self.params = p_params
        self.S = self.params['S']
        self.E = self.params['E']
        self.rho = rho

    def solve_decisions(self, c1_guess: float, w: float, r: float, X_vec: np.array, BQ_val: float,
                        enforce_no_borrowing: bool = True):
        r"""
        Given an initial consumption guess c1, calculate the full path of variables.

        :param enforce_no_borrowing: If True, clamps savings to be non-negative.
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"

        # In SS, r is constant across ages/time, but we support vector if needed
        if isinstance(r, float):
            r_vec = np.full(self.S, r)
        else:
            r_vec = r

        assert len(r_vec) == self.S, f"r_vec length {len(r_vec)} must match S {self.S}"

        # 1. Consumption Path (Euler Equation - unconstrained)
        c_vec = get_consumption_path(c1_guess, r_vec, self.rho, self.params)

        # 2. Labour Supply (Intratemporal FOC)
        n_vec = get_labour_supply(w, c_vec, self.params)

        # 3. Savings Path (Budget Constraint with borrowing constraint)
        b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_val, self.params,
                                  enforce_no_borrowing=enforce_no_borrowing)

        return c_vec, n_vec, b_vec

    def get_last_period_savings(self, c1_guess: float, w: float, r: float, X_vec: np.array, BQ_val: float) -> float:
        r"""
        Error function for the rootfinder (Shooting Method).
        Target: Savings at end of life (b_{S+1}) should be 0.

        NOTE: Uses enforce_no_borrowing=False to get the unconstrained path.
        The shooting method should find c1 such that b_s >= 0 for all s naturally.
        """
        try:
            # Get UNCONSTRAINED savings path to see true error
            _, _, b_vec = self.solve_decisions(c1_guess, w, r, X_vec, BQ_val, enforce_no_borrowing=False)
            # b_vec has length S+1. Index S is b_{S+1}.
            return b_vec[self.S]
        except (ValueError, FloatingPointError):
            # Return high penalty if path is invalid (e.g. complex numbers)
            return 1e10

    def get_objective_with_borrowing_penalty(self, c1_guess: float, w: float, r: float,
                                              X_vec: np.array, BQ_val: float) -> float:
        r"""
        Objective function that penalizes both:
        1. b_{S+1} != 0 (terminal condition)
        2. Any negative savings (borrowing violations)

        This ensures the solver finds a feasible path.
        """
        try:
            _, _, b_vec = self.solve_decisions(c1_guess, w, r, X_vec, BQ_val, enforce_no_borrowing=False)

            # Terminal condition: b_{S+1} should be 0
            terminal_error = b_vec[self.S] ** 2

            # Borrowing penalty: penalize any negative savings
            negative_savings = np.minimum(b_vec[self.E:], 0.0)
            borrowing_penalty = np.sum(negative_savings ** 2) * 100.0  # Heavy penalty

            return terminal_error + borrowing_penalty
        except (ValueError, FloatingPointError):
            return 1e10

    def solve_steady_state(self, w: float, r: float, X_vec: np.array, BQ_val: float,
                           c_init_guess_range: tuple = (1e-5, 50.0)):
        r"""
        Inner loop rootfinder algorithm that finds the optimal c_{E+1} (c1)
        such that the lifetime budget constraint holds (b_{last} = 0) AND
        savings are non-negative throughout (no borrowing).

        Strategy:
        1. Try Brent's method for root finding (faster when feasible)
        2. Check if solution satisfies borrowing constraint
        3. If not, use minimization with borrowing penalty

        Ref: algo:steady_state_solution
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"
        assert c_init_guess_range[0] < c_init_guess_range[1], "c_init_guess_range must be (low, high) with low < high"

        c_optimal = None

        # First check if the bracket contains a root
        f_low = self.get_last_period_savings(c_init_guess_range[0], w, r, X_vec, BQ_val)
        f_high = self.get_last_period_savings(c_init_guess_range[1], w, r, X_vec, BQ_val)

        # Try Brent's method if bracket contains root
        if f_low * f_high < 0:
            try:
                c_optimal = brentq(
                    self.get_last_period_savings,
                    c_init_guess_range[0],
                    c_init_guess_range[1],
                    args=(w, r, X_vec, BQ_val),
                    xtol=1e-5
                )

                # Check if this solution satisfies borrowing constraint
                _, _, b_vec = self.solve_decisions(c_optimal, w, r, X_vec, BQ_val, enforce_no_borrowing=False)
                if np.any(b_vec[self.E:] < -1e-8):
                    # Borrowing constraint violated - need to find lower c1
                    c_optimal = None
            except ValueError:
                pass

        # If Brent's failed or solution has borrowing, use minimization with penalty
        if c_optimal is None:
            result = minimize_scalar(
                self.get_objective_with_borrowing_penalty,
                bounds=c_init_guess_range,
                method='bounded',
                args=(w, r, X_vec, BQ_val),
                options={'xatol': 1e-6}
            )

            c_optimal = result.x

            # Check final solution quality
            _, _, b_vec = self.solve_decisions(c_optimal, w, r, X_vec, BQ_val, enforce_no_borrowing=False)
            terminal_error = abs(b_vec[self.S])
            min_savings = np.min(b_vec[self.E:])

            if terminal_error > 1e-3 or min_savings < -1e-3:
                import warnings
                warnings.warn(
                    f"Household solver: Suboptimal solution. "
                    f"b_{{S+1}} = {b_vec[self.S]:.4f}, min(b) = {min_savings:.4f}. "
                    f"Parameters: r={r:.4f}, BQ={BQ_val:.4f}, X_max={X_vec.max():.4f}"
                )

        # Return the final decision paths (with borrowing constraint for safety)
        return self.solve_decisions(c_optimal, w, r, X_vec, BQ_val, enforce_no_borrowing=False)