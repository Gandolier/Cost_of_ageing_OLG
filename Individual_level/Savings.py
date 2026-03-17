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
    \hat{b}_{s+1} = e^(-g_y) * [(1 - tau_l) * w * n + (1 + r_net) * \hat{b}_s + X + BQ - \hat{c}]

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

            print(
                f"age s={s}\n"
                f" (1-tau_l)={(1 - tau_l):.5g}; w={w:.5g}; n_s={n_vec[s]:.5g}; "
                    f"(1-tau_l)*w*n={(1 - tau_l) * w * n_vec[s]:.5g}"

                f" (1+r)={(1 + r_vec[s]):.5g}; b_s={b_vec[s]:.5g}; "
                    f"(1+r)*b_s={(1 + r_vec[s]) * b_vec[s]:.5g};"

                f" X_s={X_vec[s]:.5g}; BQ_val={BQ_val:.5g};"

                f" e^(-g_y)={np.exp(-g_y):.5g}; c={c_vec[s]:.5g};"
                f" b_s+1={b_next:.5g};\n"

            )

    if debug:
        # For vectors, print averages (as requested). Slices exclude non-economic ages.
        age_slice = slice(E, S)
        age_slice_next = slice(E + 1, S + 1)
        prefix = f" {debug_prefix}" if debug_prefix else ""
        print(f"[get_savings_path{prefix}] b_vec formula components (means over s={E}..{S-1})")

        print(
            f" (1-tau_l)={(1-tau_l):.5g}; w={w:.5g}; mean(n_vec)={np.mean(n_vec[age_slice]):.5g}; "
                f"(1-tau_l)*w*n={(1-tau_l)*w*np.mean(n_vec[age_slice]):.5g}"
            
            f" (1+r)={(1+np.mean(r_vec)):.5g}; mean(b_vec)={np.mean(b_vec[age_slice]):.5g}; "
                f"(1+r)*b_s={(1+np.mean(r_vec))*np.mean(b_vec[age_slice]):.5g};"
            
            f" max(X)={max(X_vec):.5g}; BQ_val={BQ_val:.5g};"
            
            f" e^(g_y)={np.exp(g_y):.5g}; e^(-g_y)={np.exp(-g_y):.5g}; mean(c_vec)={np.mean(c_vec[age_slice]):.5g};"
        )
        print(
            "  assets (means):"
            f" b_s={np.mean(b_vec[age_slice]):.6g},"
            f" b_s+1={np.mean(b_vec[age_slice_next]):.6g}"
        )

    # Return vector aligned with ages E+1 to S (decisions made)
    # Usually we want the 'stock' of assets at each age.
    # b_vec[0] is b_{E+1} (start of labour life), b_vec[1] is b_{E+2}...
    # Note: The output size and indexing depends on how Aggregate Capital sums it.
    # Usually we return the whole path including the final boundary b_{S+1}.
    return b_vec