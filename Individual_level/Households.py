import numpy as np
from scipy.optimize import brentq

from Individual_level.Consumption import get_consumption_path
from Individual_level.Labour import get_labour_supply
from Individual_level.Savings import get_savings_path
from main import reimport


class Household:
    r"""
    Represents the representative household's lifecycle problem in the Steady State.
    Coordinates Consumption, Labour, and Savings decisions.
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

    def solve_decisions(self, c1_guess: float, w: float, r: float, X_vec: np.array, BQ_val: float):
        r"""
        Given an initial consumption guess c1, calculate the full path of variables.
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
        b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_val, self.params)

        return c_vec, n_vec, b_vec

    def get_last_period_savings(self, c1_guess: float, w: float, r: float, X_vec: np.array, BQ_val: float) -> float:
        r"""
        Error function for the rootfinder (Shooting Method).
        Target: Savings at end of life (b_{S+1}) should be 0.
        """
        try:
            _, _, b_vec = self.solve_decisions(c1_guess, w, r, X_vec, BQ_val)
            # b_vec has length S+1. Index S is b_{S+1}.
            return b_vec[self.S]
        except (ValueError, FloatingPointError):
            # Return high penalty if path is invalid (e.g. complex numbers)
            return 1e10

    def solve_steady_state(self, w: float, r: float, X_vec: np.array, BQ_val: float,
                           c_init_guess_range: tuple = (1e-5, 50.0)):
        r"""
        Inner loop rootfinder algorithm that finds the optimal c_{E+1} (c1)
        such that the lifetime budget constraint holds (b_{last} = 0).

        Uses Brent's method.
        Ref: algo:steady_state_solution
        """
        assert len(X_vec) == self.S, f"X_vec length {len(X_vec)} must match S {self.S}"
        assert c_init_guess_range[0] < c_init_guess_range[1], "c_init_guess_range must be (low, high) with low < high"

        # Find root of get_last_period_savings
        # b_last(c) is monotonic in c (higher initial c -> lower final savings)

        try:
            c_optimal = brentq(
                self.get_last_period_savings,
                c_init_guess_range[0],
                c_init_guess_range[1],
                args=(w, r, X_vec, BQ_val),
                xtol=1e-5
            )
        except ValueError as e:
            # Handle root finding failures (e.g., bracket interval doesn't contain zero)
            raise ValueError(
                f"Household SS solver failed with {e}.\n"
                f"The optimization bounds are {c_init_guess_range}."
            )

        # Re-compute full path with optimal c
        return self.solve_decisions(c_optimal, w, r, X_vec, BQ_val)