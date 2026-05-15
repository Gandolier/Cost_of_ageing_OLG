"""
Stage-1 (and beyond) regression verification harness for the habits-formation refactor.

Uses the reference parameter set documented at the end of
`Documentation/habits_coding.md` (the "before any changes are implemented" set):
- demographics loaded directly from `Data/pops.csv`
- fixed g_n = -0.00568, I_total = -0.251822 / pop_base
- chi_s = np.ones(100) (no chi calibration)
- single `SteadyStateEquilibrium.solve()` call

Dumps the equilibrium r, K, L, full c_vec and (per the spec's "Required Sanity
Output") prints r to 6dp, K_hat to 4dp, and the L1 norm of c_vec.

This file is intentionally kept in the repo as a reusable regression tool for
subsequent habits stages (compare h=0 results across stages 1-6).

Usage:
    python verify_h_stage1.py [<out.npz>]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from Steady_state_equilibrium.SS_Solver import SteadyStateEquilibrium  # noqa: E402


def build_reference_inputs():
    """Build the (params, vectors, solver_params) triple from the reference set
    documented at the end of Documentation/habits_coding.md."""
    pops = pd.read_csv(REPO_ROOT / "Data/pops.csv")
    omega = pops["omega"]
    rho = pops["rho"]
    i = pops["i"]

    g_n = -0.00568
    params = {
        "A": 1.4,
        "alpha": .39,
        "delta": .05,
        "tau_c": 0.,
        "tau_l": .22,
        "tau_k": 0.,
        "R": 62,
        "E": 20,
        "S": 100,
        "g_n": g_n,
        "g_y": .04,
        "beta": .94,
        "sigma": 1.97,
        "replacement_rate": .3,
        "ltilde": 1.,
        "b_ellip": .4309,
        "upsilon": 1.7648,
        "chi_s": np.ones(100),
        "h": 0.0,
    }

    # Normalizing population to 1 for convenience
    pop_base = omega.sum()
    omega_norm = omega / pop_base
    I_norm = -0.251822 / pop_base

    vectors = {
        "omega": omega_norm.values,
        "rho": rho.values,
        "i_rate": i.values,
        "I_total": I_norm,
    }

    solver_params = {
        "r_guess": 0.05,
        "BQ_guess": 0.2,
        "tax_guess": .22,
        "policy_mode": "fix_tax",
        "tax_type": "tau_l",
        "xi": 0.1,
        "tol": 1e-2,
        "max_iter": 800,
        "debug": False,
    }
    return params, vectors, solver_params


def main(outpath):
    params, vectors, solver_params = build_reference_inputs()

    steady_state = SteadyStateEquilibrium(params, vectors)

    crash = None
    ss_results = None
    try:
        ss_results = steady_state.solve(**solver_params)
    except AssertionError as e:
        # The reference param set is known to potentially crash inside the solver
        # (the user explicitly noted "the equilibrium does not converge"). We
        # still want a deterministic regression artefact for later stages, so
        # capture the exception text as the sentinel and persist NaNs.
        crash = repr(e)

    if ss_results is None:
        r = np.float64(np.nan)
        K = np.float64(np.nan)
        L = np.float64(np.nan)
        c_vec = np.full(params["S"], np.nan, dtype=np.float64)
    else:
        r = np.float64(ss_results["r"])
        K = np.float64(ss_results["K"])
        L = np.float64(ss_results["L"])
        c_vec = np.asarray(ss_results["c_vec"], dtype=np.float64)
    c_l1 = float(np.sum(np.abs(c_vec)))

    np.savez(outpath, r=r, K=K, L=L, c_vec=c_vec)

    print(f"Saved → {outpath}")
    if crash is not None:
        print(f"NOTE: solver crashed with AssertionError: {crash}")
    # Required sanity output (habits_coding.md §"Required Sanity Output"):
    print(f"r           = {r:.6f}")
    print(f"K_hat       = {K:.4f}")
    print(f"||c_vec||_1 = {c_l1!r}")
    # Extra detail for bit-comparison logs:
    print(f"r (repr)    = {r!r}")
    print(f"K (repr)    = {K!r}")
    print(f"L (repr)    = {L!r}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "verify_post.npz"
    main(out)
