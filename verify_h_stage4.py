"""
Stage-4 regression verification for the habits-formation refactor.

Stage 4 replaces the scalar Brent forward-shoot in `Household` with a
*global* vector root-finder over c[E:S]. Per the spec's pass criterion
(`Documentation/habits_coding.md`, Stage 4) the new solver must, at
h=0, reproduce the no-habit household optimum to within 1e-5 relative
error.

The pre-habit shooter (`get_last_period_savings` in the previous version
of Households.py) defines a 1-D problem
    F(c_E) := b_{S+1}(c_E) + 100 * sum( min(0, b[E+1:S]) )
and solves F(c_E) = 0 by Brent's method. At h=0 the new global root-
finder's residual system has exactly one degree of freedom (c_E pins
down c[E:S] via the forward Euler), so its unique solution must coincide
with Brent's solution to F(c_E)=0 whenever Brent succeeds (i.e. whenever
the bracket [1e-5, 50] straddles a sign change of F).

This script builds an *in-place* reference brent shooter that mirrors
the pre-habit logic exactly (same penalty weight, same labour FOC via
M_vec = c^(-sigma) at h=0, same savings recursion) and sweeps a battery
of moderate prices (w, r, BQ_val) for which the bracket reliably
straddles a zero. At each price triple it compares the reference
(brent + penalty) and the new global solver on c, n, b.

Usage:
    python verify_h_stage4.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from Individual_level.Consumption import get_consumption_path  # noqa: E402
from Individual_level.Labour import get_labour_supply  # noqa: E402
from Individual_level.Savings import get_savings_path  # noqa: E402
from Individual_level.HabitUtility import compute_M_from_c  # noqa: E402
from Individual_level.Households import (  # noqa: E402
    Household, build_bq_receipt_vec,
)

BORROWING_PENALTY_WEIGHT = 100.0


def _ref_decisions(c1, w, r, X_vec, BQ_vec, rho, params):
    """Pre-habit-style forward-iterate (Euler) + labour FOC + savings."""
    r_vec = np.full(params["S"], r)
    c_vec = get_consumption_path(c1, r_vec, rho, params)
    M_vec = compute_M_from_c(c_vec, rho, params)  # at h=0 this is c**(-sigma)
    n_vec = get_labour_supply(w, M_vec, params)
    b_vec = get_savings_path(c_vec, n_vec, w, r_vec, X_vec, BQ_vec, params)
    return c_vec, n_vec, b_vec


def _ref_terminal(c1, w, r, X_vec, BQ_vec, rho, params):
    try:
        _, _, b_vec = _ref_decisions(c1, w, r, X_vec, BQ_vec, rho, params)
        if not np.all(np.isfinite(b_vec)):
            return 1e10
        E, S = params["E"], params["S"]
        neg = np.minimum(b_vec[E + 1:S], 0.0)
        return float(b_vec[S]) + float(np.sum(neg)) * BORROWING_PENALTY_WEIGHT
    except (ValueError, FloatingPointError, AssertionError):
        return 1e10


def ref_solve_brent(w, r, X_vec, BQ_val, omega, rho, params):
    """Pre-habit-style scalar-brent shooter with the §10.1 penalty."""
    BQ_vec = build_bq_receipt_vec(BQ_val, omega, params["E"], params["R"])
    c_low, c_high = 1e-5, 50.0
    f_low = _ref_terminal(c_low, w, r, X_vec, BQ_vec, rho, params)
    f_high = _ref_terminal(c_high, w, r, X_vec, BQ_vec, rho, params)
    if not (np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0):
        raise RuntimeError(
            f"Reference brent has no sign change at (w={w:.3g}, r={r:.3g}, "
            f"BQ={BQ_val:.3g}); pick a different price point."
        )
    c1 = brentq(_ref_terminal, c_low, c_high,
                args=(w, r, X_vec, BQ_vec, rho, params), xtol=1e-8)
    return _ref_decisions(c1, w, r, X_vec, BQ_vec, rho, params)


def _load_demography():
    """Replicates the demographic preprocessing of Tests.ipynb."""
    import scipy.interpolate as si
    omega = pd.read_excel(REPO_ROOT / "Data/total_pop.xlsx")
    omega.set_index("Year", inplace=True); omega /= 1000

    def get_mort(totpers):
        data = pd.read_excel(REPO_ROOT / "Data/death_probability.xlsx")
        data.set_index("Year", inplace=True)
        mort = pd.DataFrame(index=data.index, columns=np.arange(totpers) + 1)
        for i, year in enumerate(data.index):
            mort_data = data.iloc[i, :]
            mort_data.loc[-1] = 0.; mort_data.loc[-2] = 0.
            mort_data.loc[totpers + 1] = 1.; mort_data.loc[totpers + 2] = 1.
            age_midp = mort_data.index.values
            fit_mort = si.interp1d(age_midp, mort_data, kind="cubic",
                                   bounds_error=False, fill_value=0)
            age_model = np.linspace(100 / totpers, 100, totpers) - (0.5 * 100 / totpers)
            mort.loc[year] = fit_mort(age_model)
        return mort[(mort >= 0.) & (mort <= 1.)].ffill(axis=1)

    rho_df = get_mort(100)
    pop_base = omega.sum(axis=1)
    omega_norm = omega.div(pop_base, axis=0)
    return omega_norm.loc[2025][:-1].values, rho_df.loc[2025].values


def main():
    omega, rho = _load_demography()

    params = dict(
        A=1.4, alpha=.39, delta=.05, tau_c=0., tau_l=.22, tau_k=0.,
        R=62, E=20, S=100, g_n=-0.00568, g_y=.054, beta=.905, sigma=1.97,
        replacement_rate=.3, ltilde=1., b_ellip=.4309, upsilon=1.7648,
        chi_s=np.ones(100), h=0.0,
    )

    # Moderate-price battery (bracket [1e-5, 50] reliably straddles a
    # zero of the pre-habit shooter at these points).
    price_battery = [
        # (w,   r,    BQ_val)
        (1.5,  0.05, 0.05),
        (1.5,  0.05, 0.20),
        (2.0,  0.04, 0.30),
        (2.5,  0.03, 0.40),
        (3.0,  0.02, 0.10),
        (1.0,  0.06, 0.10),
        (4.0,  0.01, 0.50),
    ]

    overall_pass = True
    for (w, r, BQ_val) in price_battery:
        X_vec = np.zeros(params["S"])
        X_vec[params["R"]:] = 0.3 * w  # replacement_rate * w

        try:
            c_ref, n_ref, b_ref = ref_solve_brent(
                w, r, X_vec, BQ_val, omega, rho, params,
            )
        except RuntimeError as exc:
            print(f"[w={w} r={r} BQ={BQ_val}] SKIP — {exc}")
            continue

        hh = Household(params, rho)
        c_new, n_new, b_new = hh.solve_steady_state(
            w=w, r=r, X_vec=X_vec, BQ_val=BQ_val, omega=omega,
        )

        triples = [("c_vec", c_ref, c_new),
                   ("n_vec", n_ref, n_new),
                   ("b_vec", b_ref, b_new)]
        worst = 0.0
        for name, a, b in triples:
            diff = float(np.max(np.abs(a - b)))
            scale = max(float(np.max(np.abs(a))), 1e-30)
            rel = diff / scale
            worst = max(worst, rel)
        ok = worst < 1e-5
        overall_pass &= ok
        print(f"[w={w:>4} r={r:>5} BQ={BQ_val:>5}] worst rel = {worst:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print("\nOVERALL:", "PASS" if overall_pass else "FAIL")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
