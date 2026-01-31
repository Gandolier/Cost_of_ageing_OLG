import numpy as np


class Aggregator:
    r"""
    Handles the aggregation of individual household variables into
    state-level stationary variables (\hat{L}, \hat{K}, \hat{C}).
    """

    def __init__(self, omega: np.array, g_n: float = 0.0, imigr_rate: np.array = None, imigr_total: int = None,
                 E: int = 20):
        r"""
        :param omega: Stationary population distribution \hat{\omega} (normalized to sum to 1 or N_tilde)
        :param g_n: Population growth rate (\tilde{g}_n)
        :param imigr_rate: Age-specific migration rates (i_s)
        :param imigr_total: Total number of migrants (I)
        :param E: Entry age (start of economic life)
        """
        self.omega = omega
        self.g_n = g_n
        self.imigr_rate = imigr_rate if imigr_rate is not None else np.zeros_like(omega)
        self.imigr_total = imigr_total if imigr_total is not None else 0
        self.E = E

    def get_aggregate_labor(self, n_vec: np.array) -> float:
        r"""
        Calculates Aggregate Labor Supply \hat{L}.
        Ref: eq:aggregate_labour_stat
        """
        # Ensure dimensions match
        assert n_vec.shape == self.omega.shape, f"Labor vector {n_vec.shape} must match Pop vector {self.omega.shape}"

        # Sum from E+1 to S (indices E to S-1)
        # Assuming n_vec is already 0 for s < E, simple sum works, but explicit slicing is safer
        return np.sum(self.omega[self.E:] * n_vec[self.E:], axis=0)

    def get_aggregate_capital(self, b_vec: np.array) -> float:
        r"""
        Calculates Aggregate Capital Supply \hat{K}.
        Ref: eq:aggregate_capital_stat

        Assumption: b_vec[s] represents savings held by agent of age s (variable b_{s,t} in tex).
        Note: Savings b_{s,t} comes from previous period decision.

        Formula sum indices: s = E+2 to S.
        In 0-based vector indices (where index i is age i+1):
        Age E+2 corresponds to index E+1.
        Age S corresponds to index S-1.
        So slice [E+1 : S].

        Corresponding omega index is s-1.
        For age E+2 (index E+1), s-1 is index E.
        For age S (index S-1), s-1 is index S-2.
        So omega slice [E : -1].
        """
        # We shift omega to get \hat{\omega}_{s-1} (previous generation size)

        # Term 1: Survivors capital
        # \hat{\omega}_{s-1} * \hat{b}_s.
        term1 = np.sum(self.omega[self.E:-1] * b_vec[self.E + 1:-1], axis=0)

        # Term 2: Migrants capital (assuming they bring capital b_s)
        # i_s * \hat{I} * \hat{b}_s
        # i_s matches b_s index (current age)
        term2 = np.sum(self.imigr_rate[self.E + 1:] * self.imigr_total * b_vec[self.E + 1:-1], axis=0)

        return (1 / (1 + self.g_n)) * (term1 + term2)

    def get_aggregate_consumption(self, c_vec: np.array) -> float:
        r"""
        Calculates Aggregate Consumption \hat{C}.
        Ref: eq:clearing_output_stat (definition of \hat{C}_t)
        """
        assert c_vec.shape == self.omega.shape, f"Consumption vector {c_vec.shape} must match Pop vector {self.omega.shape}"

        return np.sum(self.omega[self.E:] * c_vec[self.E:], axis=0)

    def get_aggregate_bequests(self, b_vec: np.array, rho: np.array, r: float) -> float:
        r"""
        Calculates Aggregate Bequests \hat{BQ}.
        Ref: eq:bequests_stat

        Formula: \hat{BQ} = ((1+r)/(1+g_n)) * \sum \rho_{s-1} * \hat{\omega}_{s-1} * \hat{b}_s
        Sum from s=E+2 to S.
        """
        # Indices match aggregate capital logic
        # rho_{s-1} * omega_{s-1} * b_s

        # Slices:
        # s runs E+2 to S (ages) -> indices E+1 to S-1
        # s-1 runs E+1 to S-1 (ages) -> indices E to S-2

        # omega[E:-1] corresponds to indices E to S-2
        # b_vec[E+1:-1] corresponds to indices E+1 to S-1

        bequest_sum = np.sum(rho[self.E:-1] * self.omega[self.E:-1] * b_vec[self.E + 1:-1], axis=0)

        BQ = ((1 + r) / (1 + self.g_n)) * bequest_sum
        return BQ

