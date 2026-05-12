import numpy as np


def get_labour_supply(w: float, c_vec: np.array, p_params: dict) -> np.array:
    r"""
    Calculates stationary labour supply n_{s,t} given consumption path.
    Ensure inputs are arrays or broadcast correctly
    w is scalar in SS, tau_l is vector

    Ref: eq:labour_n
    n_{s,t} = l_tilde * [ 1 + ( term )^(upsilon / (upsilon - 1)) ] ^ (-1/upsilon)
    where term = (b * c^sigma) / (l_tilde * w * (1 - tau_l))
    """
    S = p_params['S']
    E = p_params['E']
    sigma = p_params['sigma']  # coefficient of relative risk aversion
    l_tilde = p_params['ltilde']  # time endowment per period
    b_ellip = p_params['b_ellip']  # \b in utility function
    upsilon = p_params['upsilon']  # \upsilon in utility function
    tau_l = p_params['tau_l']  # labour income tax rate
    chi_s = p_params['chi_s'][E:]

    n_vec = np.zeros(S)

    # Calculate only for economically active population [E:]
    # Slicing inputs
    # Note: w * (1 - tau_l) is the net wage
    # tau_l is a scalar float
    tau_l_active = tau_l

    net_wage = w * (1 - tau_l_active)

    assert np.all(net_wage > 0), "Net wage must be strictly positive for all ages"

    # Numerator: b * c^sigma
    # c_vec is 0 before E, so slice it
    c_active = c_vec[E:]
    numer = b_ellip * (c_active ** sigma) * chi_s

    # Denominator: l_tilde * net_wage
    denom = l_tilde * net_wage

    ratio = np.maximum(numer / denom, 1e-10)

    # Power term
    exponent = upsilon / (upsilon - 1)

    # Final formula
    # n = l_tilde * (1 + ratio^exponent)^(-1/upsilon)
    n_active = l_tilde * (1 + ratio ** exponent) ** (-1 / upsilon)

    n_vec[E:] = n_active

    return n_vec