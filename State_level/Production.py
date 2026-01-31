import numpy as np


class Firm:
    r"""
    Represents the representative firm behavior and production technology.
    Calculates Output, Interest Rate, and Wages based on Stationary Aggregates.
    """

    def __init__(self, A: float, alpha: float, delta: float, tau_c: float = 0.0):
        r"""
        :param A: Total Factor Productivity
        :param alpha: Capital share of output
        :param delta: Depreciation rate
        :param tau_c: Corporate tax rate
        """
        assert A > 0, f"Total Factor Productivity (A) must be positive, got {A}"
        assert 0 < alpha < 1, f"Capital share (alpha) must be between 0 and 1, got {alpha}"
        assert 0 <= delta <= 1, f"Depreciation rate (delta) must be between 0 and 1, got {delta}"
        assert 0 <= tau_c < 1, f"Corporate tax rate (tau_c) must be between 0 and 1, got {tau_c}"


        self.A = A
        self.alpha = alpha
        self.delta = delta
        self.tau_c = tau_c

    def get_output(self, K: float, L: float) -> float:
        r"""
        Cobb-Douglas Production Function (Stationary).
        Ref: eq:production_stat
        """
        assert K >= 0, f"Aggregate Capital (K) cannot be negative, got {K}"
        assert L >= 0, f"Aggregate Labor (L) cannot be negative, got {L}"

        return self.A * (K ** self.alpha) * (L ** (1 - self.alpha))

    def get_interest_rate(self, K: float, L: float) -> float:
        r"""
        Marginal Product of Capital (r).
        Ref: eq:key_rate_explicit
        """
        assert K >= 0, f"Aggregate Capital (K) cannot be negative, got {K}"
        assert L >= 0, f"Aggregate Labor (L) cannot be negative, got {L}"

        K = max(K, 1e-10)

        # Calculate MPK derived from Cobb-Douglas: alpha * A * (L/K)^(1-alpha)
        mpk = self.alpha * self.A * ((L / K) ** (1 - self.alpha))

        return (1 - self.tau_c) * (mpk - self.delta)

    def get_wage(self, K: float, L: float) -> float:
        r"""
        Marginal Product of Labor (w).
        Ref: eq:wage_stat
        """
        assert K >= 0, f"Aggregate Capital (K) cannot be negative, got {K}"
        assert L >= 0, f"Aggregate Labor (L) cannot be negative, got {L}"

        L = max(L, 1e-10)

        # Alternatively: w = (1-alpha) * A * (K/L)^alpha
        return (1 - self.alpha) * self.A * ((K / L) ** self.alpha)

    def get_wage_from_r(self, r: float) -> float:
        r"""
        Calculates wage based on interest rate (Factor Price Frontier).
        Useful for solving the SS when guessing 'r'.
        Ref: eq:combined_wage_function
        """
        # Inverse of MPK to find K/L ratio, then plug into wage eq
        term = (r / (1 - self.tau_c)) + self.delta
        inner = term / (self.A * self.alpha)

        return self.A * (1 - self.alpha) * (inner ** (self.alpha / (self.alpha - 1)))