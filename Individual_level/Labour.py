import numpy as np


def get_labour_supply(w: float, c_vec: np.array, p_params: dict) -> np.array:
    r"""
    Calculates stationary labour supply n_{s,t} given consumption path.
    Ensure inputs are arrays or broadcast correctly
    w is scalar in SS, tau_l is vector

    NOTE: Labor supply is only computed for working ages [E, R).
          Retirees (age >= R) have n=0 by definition.

    Ref: eq:labour_function_explicit
    n_{s,t} = l_tilde * [ 1 + ( term )^(upsilon / (upsilon - 1)) ] ^ (-1/upsilon)
    where term = (b * c^sigma) / (l_tilde * w * (1 - tau_l))
    """
    S = p_params['S']
    E = p_params['E']
    R = p_params['R']  # Retirement age
    sigma = p_params['sigma']  # coefficient of relative risk aversion
    l_tilde = p_params['ltilde']  # time endowment per period
    b_ellip = p_params['b_ellip']  # \b in utility function
    upsilon = p_params['upsilon']  # \upsilon in utility function
    tau_l = p_params['tau_l']  # labour income tax rate

    n_vec = np.zeros(S)

    # Calculate only for working-age population [E:R)
    # Retirees (age >= R) have n=0
    tau_l_active = tau_l

    net_wage = w * (1 - tau_l_active)

    assert np.all(net_wage > 0), "Net wage must be strictly positive for all ages"

    # Numerator: b * c^sigma
    # Only compute for working ages [E:R)
    c_working = c_vec[E:R]
    numer = b_ellip * (c_working ** sigma)

    # Denominator: l_tilde * net_wage
    denom = l_tilde * net_wage

    ratio = np.maximum(numer / denom, 1e-10)

    # Power term
    exponent = upsilon / (upsilon - 1)

    # Final formula
    # n = l_tilde * (1 + ratio^exponent)^(-1/upsilon)
    n_working = l_tilde * (1 + ratio ** exponent) ** (-1 / upsilon)

    # Only assign to working ages; retirees stay at 0
    n_vec[E:R] = n_working

    return n_vec