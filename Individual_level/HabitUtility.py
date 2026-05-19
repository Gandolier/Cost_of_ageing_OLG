import numpy as np


def compute_delta_c(c_vec: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the stationary habit-adjusted consumption:
        Delta_c[s] = c_vec[s] - h * exp(-g_y) * c_vec[s-1]
    with the boundary c_vec[E-1] = 0 (no initial habit), which the existing
    zero-initialization of c_vec[:E] already provides.

    Returns array of shape (S,) with zeros for s < E.
    """
    h = params['h']
    g_y = params['g_y']
    c_min = params.get('c_min', 0.0)
    E = params['E']
    S = params['S']

    delta_c = np.zeros(S)
    # Vectorised: delta_c[E] uses c_vec[E-1] = 0, automatic
    delta_c[E:] = c_vec[E:] - h * np.exp(-g_y) * c_vec[E-1:S-1] - c_min
    return delta_c


def compute_M_from_c(c_vec: np.ndarray, rho: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the effective marginal utility of consumption:
        M[s] = Delta_c[s]^(-sigma) - h * beta * (1-rho[s]) * exp(-sigma*g_y) * Delta_c[s+1]^(-sigma)
    for s in [E, S-2], with boundary M[S-1] = Delta_c[S-1]^(-sigma) (no future).

    Returns array of shape (S,) with zeros for s < E.
    Raises ValueError if Delta_c[s] <= 0 for any s >= E (habit too strong / bad c profile).
    """
    h = params['h']
    beta = params['beta']
    sigma = params['sigma']
    g_y = params['g_y']
    E = params['E']
    S = params['S']

    delta_c = compute_delta_c(c_vec, params)

    if np.any(delta_c[E:] <= 0):
        bad = np.where(delta_c[E:] <= 0)[0] + E
        raise ValueError(
            f"Habit-adjusted consumption non-positive at ages {bad.tolist()}; "
            f"h={h} is too large for this consumption profile."
        )

    F = np.zeros(S)
    F[E:] = delta_c[E:] ** (-sigma)

    M = np.zeros(S)
    # For s in [E, S-2]: include habit-correction term
    M[E:S-1] = F[E:S-1] - h * beta * (1 - rho[E:S-1]) * np.exp(-sigma * g_y) * F[E+1:S]
    # Boundary: no future contribution
    M[S-1] = F[S-1]
    return M


def compute_M_from_delta_c(delta_c, rho, params):
    """Same as compute_M_from_c but takes Δc directly, sidestepping
    the catastrophic cancellation when Δc is re-derived from a
    reconstructed c_vec."""
    sigma, beta, g_y = params['sigma'], params['beta'], params['g_y']
    h, E, S = params['h'], params['E'], params['S']

    if np.any(delta_c[E:] <= 0):
        bad = np.where(delta_c[E:] <= 0)[0] + E
        raise ValueError(f"Δc non-positive at ages {bad.tolist()}.")

    F = np.zeros(S)
    F[E:] = delta_c[E:] ** (-sigma)
    M = np.zeros(S)
    M[E:S-1] = F[E:S-1] - h*beta*(1-rho[E:S-1])*np.exp(-sigma*g_y)*F[E+1:S]
    M[S-1] = F[S-1]
    return M
