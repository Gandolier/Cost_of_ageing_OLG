import numpy as np


def get_consumption_path(c_start: float, r_vec: np.array, rho: np.array, p_params: dict) -> np.array:
    r"""
    Calculates the lifecycle consumption path given an initial guess c_start.
    Iterates forward using the Stationary Euler Equation.

    Ref: eq:euler_equations
    (c_s)^(-sigma) = e^(-sigma * g_y) * beta * (1 - rho_s) * (1 + r_net) * (c_{s+1})^(-sigma)
    => c_{s+1} = c_s * [ e^(-sigma * g_y) * beta * (1 - rho_s) * (1 + r_net) ] ^ (1/sigma)
    """
    S = p_params['S']           # max age of a household
    E = p_params['E']           # entry age (index E corresponds to age E+1)
    beta = p_params['beta']     # subjective discount factor
    sigma = p_params['sigma']   # coefficient of relative risk aversion
    g_y = p_params['g_y']       # rate of labour augmenting technological growth
    rho = rho                   # mortality probability
    tau_k = p_params['tau_k']   # capital income tax rate
    c_min = p_params.get('c_min', 0.0)

    assert len(rho) == S, f"rho length {len(rho)} must match number of cohorts {S}"

    c_vec = np.zeros(S)
    # c_start is consumption at the start of economic life (age E+1, index E)
    c_vec[E] = c_start

    # Pre-calculate the constant growth factor part
    # Note: rho is vector of length S. rho[s] is prob of dying at age s.
    # The term in Euler eq is (1 - rho_{s,t}).

    growth_exponent = 1.0 / sigma

    # Stationarity factor e^(-g_y)
    # The equation has e^(-sigma * g_y), raised to (1/sigma) becomes e^(-g_y)
    stat_factor = np.exp(-g_y)

    # Iterate from age E+1 (index E) to S-1
    for s in range(E, S - 1):
        # Net interest rate at t+1 (perfect foresight, r is constant in SS)
        # Using r[s] assuming r is age-independent in SS, or r[s] is the rate faced at age s
        r_net = r_vec[s] * (1 - tau_k)

        # Effective discount factor including survival probability
        # beta * (1 - rho[s])
        disc_surv = beta * (1 - rho[s])

        # Calculate next period consumption
        # c_{s+1} = c_s * (beta * (1-rho) * (1+r_net))^(1/sigma) * e^(-g_y)
        growth_rate = (disc_surv * (1 + r_net)) ** growth_exponent

        c_vec[s + 1] = c_min + (c_vec[s] - c_min) * growth_rate * stat_factor

    return c_vec