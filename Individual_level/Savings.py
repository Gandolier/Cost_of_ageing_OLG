import numpy as np


def get_savings_path(
        c_vec: np.array,
        n_vec: np.array,
        w: float,
        r_vec: np.array,
        X_vec: np.array,
        BQ_val: float,
        p_params: dict,
        b_init: float = 0.0,
        clamp_to_zero: bool = False,
        debug: bool = False,
        debug_prefix: str = ""
) -> np.array:
    r"""
    Calculates the lifecycle savings path b_{s+1}.
    BQ_val is the stationary value of BQ. Note: BQ is distributed per capita.

    Ref: eq:budget_constraint_stat
    e^g_y * \hat{b}_{s+1} = (1 - tau_l) * w * n + (1 + r_net) * \hat{b}_s + X + BQ - \hat{c}
    => \hat{b}_{s+1} = e^(-g_y) * [ Income + Assets - Consumption ]

    :param clamp_to_zero: If True, enforces no-borrowing constraint by clamping
                          savings to max(0, computed_value).
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

    # Optional debug accumulators (store per-age components of the b_next formula)
    if debug:
        T = S - E
        labour_inc_vec = np.empty(T)
        r_net_vec = np.empty(T)
        capital_inc_vec = np.empty(T)
        resources_vec = np.empty(T)
        b_next_vec = np.empty(T)

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

        # Optionally enforce no-borrowing constraint
        if clamp_to_zero:
            b_next = max(0.0, b_next)

        b_vec[s + 1] = b_next

        if debug:
            j = s - E
            labour_inc_vec[j] = labour_inc
            r_net_vec[j] = r_net
            capital_inc_vec[j] = capital_inc
            resources_vec[j] = resources
            b_next_vec[j] = b_next

    if debug:
        # For vectors, print averages (as requested). Slices exclude non-economic ages.
        age_slice = slice(E, S)
        age_slice_next = slice(E + 1, S + 1)
        prefix = f" {debug_prefix}" if debug_prefix else ""
        print(f"[get_savings_path{prefix}] b_vec formula components (means over s={E}..{S-1})")
        print(
            "  scalars:"
            f" w={w:.6g}, tau_l={tau_l:.6g}, tau_k={tau_k:.6g}, g_y={g_y:.6g}, "
            f"stat_factor={stat_factor:.6g}, BQ_val={BQ_val:.6g}, b_init={b_init:.6g}, "
            f"clamp_to_zero={clamp_to_zero}"
        )
        print(
            "  inputs (means):"
            f" n_vec={float(np.mean(n_vec[age_slice])):.6g},"
            f" c_vec={float(np.mean(c_vec[age_slice])):.6g},"
            f" r_vec={float(np.mean(r_vec[age_slice])):.6g},"
            f" X_vec={float(np.mean(X_vec[age_slice])):.6g}"
        )
        print(
            "  components (means):"
            f" labour_inc={float(np.mean(labour_inc_vec)):.6g},"
            f" r_net={float(np.mean(r_net_vec)):.6g},"
            f" capital_inc={float(np.mean(capital_inc_vec)):.6g},"
            f" resources={float(np.mean(resources_vec)):.6g},"
            f" b_next={float(np.mean(b_next_vec)):.6g}"
        )
        print(
            "  assets (means):"
            f" b_s={float(np.mean(b_vec[age_slice])):.6g},"
            f" b_s+1={float(np.mean(b_vec[age_slice_next])):.6g}"
        )

    # Return vector aligned with ages E+1 to S (decisions made)
    # Usually we want the 'stock' of assets at each age.
    # b_vec[0] is b_{E+1} (start of labour life), b_vec[1] is b_{E+2}...
    # Note: The output size and indexing depends on how Aggregate Capital sums it.
    # Usually we return the whole path including the final boundary b_{S+1}.
    return b_vec