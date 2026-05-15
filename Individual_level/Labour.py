import numpy as np


def get_labour_supply(w: float, M_vec: np.ndarray, p_params: dict) -> np.ndarray:
    r"""
    Calculates stationary labour supply n_{s,t} given the effective marginal
    utility of consumption M_vec.

    Ref: habits_derivations.md §5.5 (replaces paper eq. 2.28)
        n_s = l_tilde * [ 1 + ( chi_s * b / (l_tilde * (1 - tau_l) * w * M_s) )
                            ^( upsilon / (upsilon - 1) ) ] ^ ( -1 / upsilon )

    At h = 0, M_s = c_s^(-sigma) and this reduces to the original no-habit FOC.

    Inputs:
        w        : stationary wage (scalar in SS)
        M_vec    : effective marginal utility, length S, zeros for s < E
        p_params : household parameter dict (S, E, ltilde, b_ellip, upsilon, tau_l, chi_s)
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
