import numpy as np
import scipy.optimize as opt
from scipy.interpolate import PchipInterpolator

from Steady_state_equilibrium.SS_Solver import SteadyStateEquilibrium


def get_sumsq(ellip_params, *args):
    b, upsilon = ellip_params
    elast_Frisch, nvec, ltilde = args

    mu_cfe = nvec ** (1 / elast_Frisch)

    mu_ellip = ((b / ltilde) * ((nvec / ltilde) ** (upsilon - 1)) * ((1 - ((nvec / ltilde) ** upsilon)) ** \
                                                                     ((1 - upsilon) / upsilon)))

    sumsq = ((mu_cfe - mu_ellip) ** 2).sum()

    return sumsq

def fit_ellip(ellip_init, elast_Frisch, ltilde):
    nvec = np.linspace(0.05, 0.95, 1000)
    args = (elast_Frisch, nvec, ltilde)
    bnds_elp = ((1e-12, None), (1 + 1e-12, None))

    ellip_params = opt.minimize(
        get_sumsq, ellip_init, args=(args), method='L-BFGS-B',
        bounds=bnds_elp)

    if ellip_params.success:
        b_ellip, upsilon = ellip_params.x
    else:
        raise ValueError("Failed to minimize sum of squares")
    return b_ellip, upsilon


def update_chi_from_foc(c_vec, w, tau_l, n_target, rho, params):
    """
    Closed-form inverse of the habit-modified labor FOC:
        chi_s = [l_tilde * (1 - tau_l) * w * M_s / b]
                * (eta_s^(-upsilon) - 1)^((upsilon - 1) / upsilon)
    where eta_s = n_target_s / l_tilde and M_s is the habit-adjusted
    effective marginal utility of consumption (see HabitUtility.py).

    At h=0, M_s = c_s^(-sigma) and the formula reduces to the original.
    """
    from Individual_level.HabitUtility import compute_M_from_c

    upsilon = params['upsilon']
    b = params['b_ellip']
    l_tilde = params['ltilde']
    E = params['E']
    S = params['S']

    chi_s = np.ones(S)  # default 1 for s < E (these ages don't supply labor anyway)

    # Working-age normalised labour share, with numerical guards against the
    # degenerate corner solutions n=0 and n=l_tilde
    eta = np.clip(n_target[E:] / l_tilde, 1e-6, 1 - 1e-6)

    # The leisure-bracket from the FOC, raised to (upsilon-1)/upsilon
    leisure_term = (eta ** (-upsilon) - 1) ** ((upsilon - 1) / upsilon)

    # Habit-formation effective marginal utility
    M_vec = compute_M_from_c(c_vec, rho, params)

    # Direct closed-form
    chi_s[E:] = (l_tilde * (1 - tau_l) * w * leisure_term * M_vec[E:]) / b

    return chi_s


def update_beta_from_euler(c_target, r, rho, params):
    """
    Backward-recursion inverse of the habit Euler equation.

    Computes the age-specific beta_s that reproduces the target
    consumption profile c_target at interest rate r.

    The recursion uses the identity

        F_s = beta_s (1-rho_s) e^{-σ g_y} [ (1+r_net) M_{s+1} + h F_{s+1} ]

    with terminal condition M_{S-1} = F_{S-1}.
    """
    from Individual_level.HabitUtility import compute_delta_c

    sigma = params['sigma']
    g_y = params['g_y']
    h = params['h']
    tau_k = params['tau_k']
    E = params['E']
    S = params['S']

    r_net = r * (1 - tau_k)
    edisc = np.exp(-sigma * g_y)

    delta_c = compute_delta_c(c_target, params)

    F = np.zeros(S)
    F[E:] = np.maximum(delta_c[E:], 1e-12) ** (-sigma)

    beta_s = np.full(S, 0.97)
    M = np.zeros(S)

    # terminal condition
    M[S-1] = F[S-1]

    for s in range(S-2, E-1, -1):
        denom = (1 - rho[s]) * edisc * ((1 + r_net) * M[s+1] + h * F[s+1])
        beta_s[s] = F[s] / denom
        M[s] = beta_s[s] * (1 - rho[s]) * (1 + r_net) * edisc * M[s+1]

    return beta_s


def calibrate_chi_beta(params, vectors, n_target, c_target,
                       max_iter=50, tol=1e-2, xi_chi=0.5, xi_beta=0.5,
                       ss_solve_kwargs=None):
    """
    Dummy simultaneous calibration loop for chi_s and beta_s.

    Each iteration:
      1. Solve SS with current (chi_s, beta_s)
      2. Update chi_s from labour FOC inverse
      3. Update beta_s from Euler inverse using c_target
      4. Damp updates until convergence
    """

    S = params['S']
    E = params['E']

    params = dict(params)

    chi_s = params.get('chi_s', np.ones(S)).copy()
    beta_s = params.get('beta_s', params.get('beta', 0.97) * np.ones(S)).copy()

    params['chi_s'] = chi_s
    params['beta_s'] = beta_s

    result = None

    for k in range(max_iter):

        ss = SteadyStateEquilibrium(params, vectors)
        result = ss.solve(**(ss_solve_kwargs or {}))

        if result is None:
            raise RuntimeError(f"calib iter {k}: SS failed")

        r = result['r']
        w = result['w']

        chi_s_new = update_chi_from_foc(
            c_vec=result['c_vec'],
            w=w,
            tau_l=result['tau_l'],
            n_target=n_target,
            rho=vectors['rho'],
            params=params
        )

        beta_s_new = update_beta_from_euler(
            c_target=c_target,
            r=r,
            rho=vectors['rho'],
            params=params
        )

        err_chi = np.max(np.abs(chi_s_new[E:] - chi_s[E:]) /
                         np.maximum(np.abs(chi_s[E:]), 1.0))

        err_beta = np.max(np.abs(beta_s_new[E:] - beta_s[E:]) /
                          np.maximum(np.abs(beta_s[E:]), 1e-3))

        print(f"Outer iter {k}: dchi={err_chi:.3e}, dbeta={err_beta:.3e}, r={r:.4f}, w={w:.4f}")
        print(25*"---")

        if max(err_chi, err_beta) < tol:
            params['chi_s'] = chi_s_new
            params['beta_s'] = beta_s_new
            return chi_s_new, beta_s_new, result

        chi_s = xi_chi * chi_s_new + (1 - xi_chi) * chi_s
        beta_s = xi_beta * beta_s_new + (1 - xi_beta) * beta_s

        params['chi_s'] = chi_s
        params['beta_s'] = beta_s

        if ss_solve_kwargs is not None:
            ss_solve_kwargs['r_guess'] = r
            ss_solve_kwargs['BQ_guess'] = result['BQ']

    return chi_s, beta_s, result


def calibrate_chi(params, vectors, n_target,
                  max_iter=50, tol=1e-3, xi_chi=0.99,
                  ss_solve_kwargs=None):
    """
    Outer calibration loop. At each iteration:
      1. Solve the full SS given the current chi_s (passed via params).
      2. Read off equilibrium c_vec, w, tau_l.
      3. Update chi_s from the FOC inverse.
      4. Check convergence on chi_s.

    Returns (chi_s_calibrated, final_ss_result).
    """
    if ss_solve_kwargs is None:
        ss_solve_kwargs = dict(r_guess=0.10, BQ_guess=0.05, tax_guess=0.10,
                               policy_mode='fix_pension', tax_type='tau_l',
                               xi=0.2, tol=1e-5, max_iter=300)

    S = params['S']
    R = params['R']
    chi_s = np.ones(S)  # initial guess: paper's original specification
    params = dict(params)  # local copy so we can write chi_s in
    params['chi_s'] = chi_s
    params.setdefault('h', 0.0)  # backward-compatibility: no habit formation by default

    final_result = None

    for k in range(max_iter):
        # Step 1: full SS solve given current chi_s
        ss = SteadyStateEquilibrium(params, vectors)
        result = ss.solve(**ss_solve_kwargs)

        if result is None:
            raise RuntimeError(f"Calibration iter {k}: SS failed to converge.")

        # Sync c_min with the equilibrium wage so update_chi_from_foc
        # uses the same c_min that the SS solve converged with.
        #params['c_min'] = params.get('c_min_wage_share', 0.44) * result['w']

        # Step 2-3: closed-form chi_s update
        chi_s_new = update_chi_from_foc(
            c_vec=result['c_vec'],
            w=result['w'],
            tau_l=result['tau_l'],
            n_target=n_target,
            rho=vectors['rho'],
            params=params,
        )
        # Clamp disutility of labour sensitivity
        #chi_s_new = np.clip(chi_s_new, 1e-8, 1e12)

        # Step 4: convergence check on chi_s
        denom = np.maximum(np.abs(chi_s[params['E']:]), 1.0)
        err = np.max(np.abs(chi_s_new[params['E']:] - chi_s[params['E']:]) / denom)
        print(f"Chi-iter {k}: max |dchi| = {err:.3e}, "
              f"r = {result['r']:.4f}, w = {result['w']:.4f}")
        print(25*"---")

        if err < tol:
            params['chi_s'] = chi_s_new
            return chi_s_new, result

        # Damped update — full step (xi_chi=1) tends to work but damp if oscillating
        chi_s = xi_chi * chi_s_new + (1 - xi_chi) * chi_s
        params['chi_s'] = chi_s

        # Update SS solve kwargs with equilibrium values to avoid re-solving
        ss_solve_kwargs['r_guess'] = result['r']
        ss_solve_kwargs['BQ_guess'] = result['BQ']

        final_result = result

    print(f"Calibration did not fully converge in {max_iter} iterations.")
    return chi_s, final_result


def build_target_labour_profile(
        ep_by_bin,
        hours_by_bin=None,
        constant_hours=38.0,
        h_max=100.0,
        E=20,
        S=100,
        tail_start_age=75,
        tail_zero_age=95,
        left_anchor_offset=5,
        return_components=False,
):
    """
    Construct n_target_s = (E/P)_s * h_s / h_max on a single-year grid in [0, S).

    Boundary handling is done via sentinel anchor points augmenting the binned
    data, so a single PCHIP interpolation produces the full profile with smooth
    tails. The right-tail anchor is zero for E/P (decays to zero) and
    last-observed-value for hours (held flat — workers past 75 still work).

    Parameters
    ----------
    ep_by_bin : dict[(lo, hi), float]
        Employment-to-population ratio per age bin. Use (lo, 999) or similar
        for the open-ended top bin.
        Example: {(15,19): 0.21, (20,24): 0.65, ..., (60,69): 0.32, (70, 999): 0.06}
    hours_by_bin : dict[(lo, hi), float] or None
        Mean weekly hours per worker per age bin. Bins may differ from ep_by_bin.
        If None, uses constant_hours for every age.
    constant_hours : float
        Fallback hours value used only if hours_by_bin is None.
    h_max : float
        Time-endowment normalisation, conventionally 100 hours/week.
    E, S : int
        Entry age into the labour force (exclusive below) and maximum model age.
    tail_start_age : int
        Anchor age assigned to the midpoint of the open-ended top bin.
    tail_zero_age : int
        Age by which the extrapolated E/P profile decays effectively to zero.
    left_anchor_offset : int
        Offset from tail_start_age to the left anchor age, typically 0.
    return_components : bool
        If True, returns (n_target, ep_profile, hours_profile) for diagnostics.

    Returns
    -------
    np.ndarray of shape (S,)
        n_target indexed by integer age 0..S-1. Zero below E by construction.
        If return_components=True, returns a tuple including the intermediate
        E/P and hours profiles.
    """
    ages = np.arange(S)

    def _interp_from_bins(bin_dict, right_tail_zero):
        """Interpolate via PCHIP after augmenting with sentinel anchors.

        right_tail_zero=True: anchor the right tail at zero (suited to E/P).
        right_tail_zero=False: anchor the right tail at the last observed
        value (suited to hours-conditional-on-employment).
        """
        # Pull raw midpoints and values from the binned dict
        raw_mids, raw_vals = [], []
        for (lo, hi), v in sorted(bin_dict.items()):
            mid = tail_start_age if hi >= 100 else (lo + hi) / 2.0
            raw_mids.append(mid)
            raw_vals.append(v)

        # Augment with anchor points; PCHIP will smoothly interpolate across them
        mids = [max(0, raw_mids[0] - left_anchor_offset)] + raw_mids
        vals = [0.0] + raw_vals

        if right_tail_zero:
            mids.append(tail_zero_age)
            vals.append(0.0)
            # Pin everything past tail_zero_age at zero explicitly so PCHIP
            # extrapolation never produces spurious oscillations
            if tail_zero_age < S - 1:
                mids.append(S - 1)
                vals.append(0.0)
        else:
            # Hold flat at the last observed value to the end of the age range
            mids.append(S - 1)
            vals.append(raw_vals[-1])

        interp = PchipInterpolator(np.asarray(mids, dtype=float),
                                   np.asarray(vals, dtype=float))
        out = interp(ages)
        return np.maximum(out, 0.0)  # clamp tiny negative artefacts

    ep_profile = _interp_from_bins(ep_by_bin, right_tail_zero=True)

    if hours_by_bin is not None:
        hours_profile = _interp_from_bins(hours_by_bin, right_tail_zero=False)
    else:
        hours_profile = np.full(S, constant_hours, dtype=float)

    n_target = ep_profile * hours_profile / h_max
    n_target[:E] = 0.0
    n_target = np.clip(n_target, 0.0, 1.0)

    if return_components:
        return n_target, ep_profile, hours_profile
    return n_target


def build_cons_profile(raw_mids=None, raw_vals=None, S=100, E=20):
    from scipy.interpolate import PchipInterpolator
    """Interpolate via PCHIP after augmenting with sentinel anchors.

    right_tail_zero=True: anchor the right tail at zero (suited to E/P).
    right_tail_zero=False: anchor the right tail at the last observed
    value (suited to hours-conditional-on-employment).
    """
    full_s = np.arange(S)

    if raw_mids is None:
        raw_mids = [17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 85]
    if raw_vals is None:
        raw_vals = [23.8, 36.8, 47.2, 51.6, 53., 51.3, 47., 42.5, 37.4, 31.7, 27.4, 25., 23.9, 24.6, 25.1]

    # Augment with anchor points; PCHIP will smoothly interpolate across them
    mids = [raw_mids[0]-1] + raw_mids
    vals = [raw_vals[0]] + raw_vals

    # Hold flat at the last observed value to the end of the age range
    mids.append(90)
    vals.append(raw_vals[-1])

    interp = PchipInterpolator(np.asarray(mids, dtype=float),
                                np.asarray(vals, dtype=float))
    out = interp(full_s)
    out[:E] = 0.
    return np.maximum(out, 0.0)  # clamp tiny negative artefacts