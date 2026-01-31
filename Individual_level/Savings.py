import numpy as np


def get_savings_path(
        c_vec: np.array,
        n_vec: np.array,
        w: float,
        r_vec: np.array,
        X_vec: np.array,
        BQ_val: float,
        p_params: dict,
        b_init: float = 0.0
) -> np.array:
    r"""
    Calculates the lifecycle savings path b_{s+1}.
    BQ_val is the stationary value of BQ. Note: BQ is distributed per capita.

    Ref: eq:budget_constraint_stat
    e^g_y * \hat{b}_{s+1} = (1 - tau_l) * w * n + (1 + r_net) * \hat{b}_s + X + BQ - \hat{c}
    => \hat{b}_{s+1} = e^(-g_y) * [ Income + Assets - Consumption ]
    """
    S = p_params['S']                  # max age of a household
    E = p_params['E']
    tau_l = p_params['tau_l']       # labour income tax rate
    tau_k = p_params['tau_k']       # capital income tax rate
    g_y = p_params['g_y']           # rate of labour augmenting technological growth

    assert len(n_vec) == S, f"n_vec length {len(n_vec)} must match number of cohorts {S}"
    assert len(r_vec) == S, f"r_vec length {len(r_vec)} must match number of cohorts {S}"
    assert len(X_vec) == S, f"X_vec length {len(X_vec)} must match number of cohorts {S}"

    b_vec = np.zeros(S + 1)  # Includes age E (initial) to S+1 (end of life)
    b_vec[E] = b_init

    stat_factor = np.exp(-g_y)

    # Iterate through ages s to find b_{s+1}
    # Starting from E (first economic period) up to S-1
    for s in range(E, S):
        # Labor Income: (1 - tau_l) * w * n
        labour_inc = (1 - tau_l) * w * n_vec[s]

        # Capital Income + Principal: (1 + r * (1 - tau_k)) * b_s
        r_net = r_vec[s] * (1 - tau_k)
        capital_inc = (1 + r_net) * b_vec[s]

        # Total Resources
        # In tex eq:budget_constraint: + BQ_t / N_tilde_t.
        # We assume BQ_val passed here is already normalized (\hat{BQ}).
        resources = labour_inc + capital_inc + X_vec[s] + BQ_val

        # Next period savings (Stationary transformation)
        b_next = (resources - c_vec[s]) * stat_factor

        b_vec[s + 1] = b_next

    # Return vector aligned with ages E+1 to S (decisions made)
    # Usually we want the 'stock' of assets at each age.
    # b_vec[0] is b_{E+1} (start of labour life), b_vec[1] is b_{E+2}...
    # Note: The output size and indexing depends on how Aggregate Capital sums it.
    # Usually we return the whole path including the final boundary b_{S+1}.
    return b_vec