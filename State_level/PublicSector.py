import numpy as np


class Government:
    r"""
    Handles fiscal policy, specifically pension benefits (\hat{X}) and tax rates (\tau).
    Aligned with the government budget constraint.
    """

    def __init__(self, tau_l: float, tau_k: float, tau_c: float, retirement_age: int, entry_age: int):
        r"""
        :param tau_l: Labor income tax rate
        :param tau_k: Capital income tax rate
        :param tau_c: Corporate tax rate
        :param retirement_age: The real age at which agents retire (R)
        :param entry_age: The real age at which agents enter the model (E)
        """
        assert retirement_age > entry_age, f"Retirement age ({retirement_age}) must be greater than entry age ({entry_age})"
        assert 0 <= tau_l < 1, f"Labor tax rate must be [0, 1), got {tau_l}"
        assert 0 <= tau_k < 1, f"Capital gain tax rate must be [0, 1), got {tau_k}"
        assert 0 <= tau_c < 1, f"Corporate profit tax rate must be [0, 1), got {tau_c}"

        self.tau_l = tau_l
        self.tau_k = tau_k
        self.tau_c = tau_c

        # Calculate the index corresponding to retirement age
        # Arrays are 0-indexed starting at age 1.
        # Age R corresponds to index R-1.
        # But we want index for loop "from R+1 to S".
        # If R=65, retiree start at 66.
        # Index for 66 is 65.
        # So we can just use retirement_age as the start index for slicing if retirement_age is the integer age.
        # Example: R=65. slice[65:] gives elements 65, 66... which are ages 66, 67...
        # This matches "sum s=R+1 to S"
        self.R_idx = retirement_age
        self.E_idx = entry_age

    def get_total_tax_revenue(
            self,
            omega: np.array,
            w: float,
            r: float,
            n_vec: np.array,
            b_vec: np.array,
            Y: float,
            L: float,
            K: float,
            delta: float
    ) -> float:
        r"""
        Calculates total tax revenue available for distribution.
        Used as numerator in eq:tax_rates_stat or eq:pension_payment_function.
        Ref: eq:tax_rates_functions
        """
        assert omega.ndim == 1, "Population distribution omega must be 1D array"
        assert n_vec.shape == omega.shape, f"Labor supply n_vec {n_vec.shape} must match omega {omega.shape}"
        assert len(b_vec) == len(omega) + 1, f"Savings b_vec {len(b_vec)} must be size S+1 (omega {len(omega)})"
        assert self.R_idx < len(omega), f"Retirement index {self.R_idx} exceeds model lifespan {len(omega)}"

        # Labor Tax Revenue (from the entire population E+1 to S)
        # \sum \hat{\omega} * \tau^l * w * n
        # Slicing from E to ensure we don't pick up garbage (though n should be 0)
        labour_tax = np.sum(omega[self.E_idx:] * self.tau_l * w * n_vec[self.E_idx:])

        # Capital Tax Revenue (applies to all ages s=E+1 to S)
        # \sum \hat{\omega} * \tau^k * r * b
        # b_vec here is size S+1. b_vec[s] is assets at age s+1.
        # We need assets at age s. This corresponds to index s-1.
        # For range E+1 to S (indices E to S-1 in omega), we need b indices E to S-1.
        # This is the slice b_vec[E:-1].
        savings_tax = np.sum(omega[self.E_idx:] * self.tau_k * r * b_vec[self.E_idx:-1])

        # Corporate/Profit Tax Revenue
        # \tau^c * (Y - wL - \delta K)
        profit_tax = self.tau_c * (Y - w * L - delta * K)

        return labour_tax + savings_tax + profit_tax

    def get_pension_benefits(self,
                             total_revenue: float,
                             omega: np.array) -> float:
        r"""
        Calculates stationary pension payment per retiree \hat{X}.
        Ref: eq:pension_payment_function (Stationary version)

        \hat{X} = Total Revenue / \hat{N}_R
        """
        assert omega.ndim == 1, "Population distribution omega must be 1D array"
        assert self.R_idx < len(omega), f"Retirement index {self.R_idx} exceeds model lifespan {len(omega)}"

        # Number of retirees \hat{N}_R
        # Sum population from retirement index onwards
        N_R = np.sum(omega[self.R_idx:])

        # Avoid division by zero
        if N_R == 0:
            return 0.0

        X = total_revenue / N_R
        return X

    def update_tax_rate(self,
                        target_X: float,
                        omega: np.array,
                        w: float,
                        n_vec: np.array) -> float:
        r"""
        Inverse calculation: Finds required \tau^l to fund a specific pension level X.
        Useful for the 'tau_l' iteration in the SS algorithm.
        Ref: eq:tax_rates_stat

        \hat{\tau}^l = (\hat{X} * \hat{N}_R - OtherTaxes) / LaborBase
        (Simplified assuming we solve for just one tax rate holding others constant)
        """
        assert omega.ndim == 1, "Population distribution omega must be 1D array"
        assert n_vec.shape == omega.shape, f"Labor supply n_vec {n_vec.shape} must match omega {omega.shape}"

        # Number of retirees
        N_R = np.sum(omega[self.R_idx:])

        # Total revenue required to fund pensions at level target_X
        required_revenue = target_X * N_R

        # The tax base for labor (working age population)
        # Assuming tax is paid by all workers E+1 to S.
        labor_base = np.sum(omega[self.E_idx:] * w * n_vec[self.E_idx:])

        if labor_base == 0:
            return self.tau_l  # Return current if base is 0 to avoid crash

        # Simplified: assuming this tax covers the whole burden
        new_tau = required_revenue / labor_base
        return new_tau

    def get_required_tau_l(self, target_X: float, omega: np.array, w: float, n_vec: np.array) -> float:
        r"""
        Calculates the labor income tax rate required to fund pension level X,
        assuming other taxes are zero/fixed.
        We assume, realistically, that pension tax is imposed on all workers, including the working retirees.
        That is because according to Russian law there is no exemption of retirees from pension taxes.

        Ref: eq:tax_rates_stat
        """
        N_R = np.sum(omega[self.R_idx:])
        required_revenue = target_X * N_R
        labor_base = np.sum(omega[self.E_idx:] * w * n_vec[self.E_idx:])

        if labor_base <= 1e-9:
            return 1.0

        return required_revenue / labor_base

    def get_required_tau_k(self, target_X: float, omega: np.array, r: float, b_vec: np.array) -> float:
        r"""
        Calculates the capital income tax rate required to fund pension level X.
        Ref: eq:tax_rates_stat
        """
        N_R = np.sum(omega[self.R_idx:])
        required_revenue = target_X * N_R

        # Base: sum(omega * r * b)
        # Note: Capital tax is on the interest income r*b
        # Use E_idx to start summing from start of economic life
        # b_vec is S+1, align with omega (S) via slicing [:-1]
        capital_base = np.sum(omega[self.E_idx:] * r * b_vec[self.E_idx:-1])

        if capital_base <= 1e-9:
            return 1.0

        return required_revenue / capital_base

    def get_required_tau_c(self, target_X: float, omega: np.array, Y: float, w: float, L: float, K: float,
                           delta: float) -> float:
        r"""
        Calculates the corporate profit tax rate required to fund pension level X.
        Ref: eq:tax_rates_stat
        """
        N_R = np.sum(omega[self.R_idx:])
        required_revenue = target_X * N_R

        profit_base = Y - w * L - delta * K

        if profit_base <= 1e-9:
            return 1.0

        return required_revenue / profit_base