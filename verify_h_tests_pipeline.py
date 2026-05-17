"""
Headless replication of the Tests.ipynb pipeline for h=0 regression checks
across the habits-formation refactor stages. Produces an .npz with the final
SS r, K, L, c_vec, n_vec, b_vec, and the calibrated chi_s.

This pipeline does converge (unlike the reference-set in verify_h_stage1.py),
so it is the authoritative bit-identity check for Stages 1-3 at h=0.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.interpolate as si

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from Steady_state_equilibrium.SS_Solver import SteadyStateEquilibrium  # noqa: E402
from Calibration.Labour_func import build_target_labour_profile, calibrate_chi  # noqa: E402


def get_mort(totpers):
    data = pd.read_excel(REPO_ROOT / "Data/death_probability.xlsx")
    data.set_index("Year", inplace=True)
    mort = pd.DataFrame(index=data.index, columns=np.arange(totpers) + 1)
    for i, year in enumerate(data.index):
        mort_data = data.iloc[i, :]
        mort_data.loc[-1] = 0.
        mort_data.loc[-2] = 0.
        mort_data.loc[totpers + 1] = 1.
        mort_data.loc[totpers + 2] = 1.
        age_midp = mort_data.index.values
        fit_mort = si.interp1d(age_midp, mort_data, kind="cubic",
                               bounds_error=False, fill_value=0)
        age_model = np.linspace(100 / totpers, 100, totpers) - (0.5 * 100 / totpers)
        mort.loc[year] = fit_mort(age_model)
    return mort[(mort >= 0.) & (mort <= 1.)].ffill(axis=1)


def get_tuple_bins(strings):
    groups = []
    for s in strings:
        s = s.replace(" ", "")
        if len(s) <= 5:
            groups.append((int(s[:2]), int(s[3:])))
        else:
            groups.append((int(s[:2]), 999))
    return groups


def main(outpath):
    omega = pd.read_excel(REPO_ROOT / "Data/total_pop.xlsx")
    omega.set_index("Year", inplace=True)
    omega /= 1000
    rho = get_mort(100)
    I = pd.read_excel(REPO_ROOT / "Data/age specific migration.xlsx",
                      sheet_name="migration").set_index("Year")
    I /= 1000
    i = pd.read_excel(REPO_ROOT / "Data/age specific migration.xlsx",
                      sheet_name="rate").set_index("age")["average"]

    employment = pd.read_excel(REPO_ROOT / "Data/employment.xlsx",
                               header=1).iloc[-1, 2:]
    employ = {grp: val / 100 for grp, val in
              zip(get_tuple_bins(employment.index), employment.values)}
    hours = pd.read_excel(REPO_ROOT / "Data/hours.xlsx", header=2)
    hours = {grp: val for grp, val in
             zip(get_tuple_bins(hours.iloc[1:, 0]), hours.iloc[1:, -1])}

    n_target, ep_profile, hours_profile = build_target_labour_profile(
        ep_by_bin=employ, hours_by_bin=hours, h_max=100, E=20, S=100,
        tail_start_age=70, tail_zero_age=85, left_anchor_offset=6,
        return_components=True,
    )
    hours_profile[:15] = 0
    ep_profile[:12] = 0

    g_n = omega.loc[2025].sum() / omega.loc[2024].sum() - 1

    pop_base = omega.sum(axis=1)
    omega_norm = omega.div(pop_base, axis=0)
    I_norm = I.div(pop_base, axis=0)

    vectors = {
        "omega": omega_norm.loc[2025][:-1].values,
        "rho": rho.loc[2025].values,
        "i_rate": i[:-1].values,
        "I_total": I_norm.loc[2025].values,
    }

    # ---- calibrate chi_s ----
    params = {
        "A": 1.4, "alpha": .39, "delta": .05,
        "tau_c": 0., "tau_l": .22, "tau_k": 0.,
        "R": 62, "E": 20, "S": 100,
        "g_n": g_n, "g_y": .054, "beta": .905, "sigma": 1.97,
        "replacement_rate": .3, "ltilde": 1.,
        "b_ellip": .4309, "upsilon": 1.7648,
        "chi_s": np.ones(100), "h": 0.0,
    }
    solver_params = {
        "r_guess": 0.05, "BQ_guess": 0.2, "tax_guess": .22,
        "policy_mode": "fix_tax", "tax_type": "tau_l",
        "xi": 0.1, "tol": 1e-3, "max_iter": 1000, "debug": True,
    }
    chi_s, _ = calibrate_chi(
        params=params, vectors=vectors, n_target=n_target,
        max_iter=100, tol=1e+7, xi_chi=0.9,
        ss_solve_kwargs=solver_params,
    )

    # ---- final SS solve with ----
    params["chi_s"] = chi_s

    steady_state = SteadyStateEquilibrium(params, vectors)
    ss = steady_state.solve(**solver_params)

    r = np.float64(ss["r"])
    K = np.float64(ss["K"])
    L = np.float64(ss["L"])
    c_vec = np.asarray(ss["c_vec"], dtype=np.float64)
    n_vec = np.asarray(ss["n_vec"], dtype=np.float64)
    b_vec = np.asarray(ss["b_vec"], dtype=np.float64)

    np.savez(outpath, r=r, K=K, L=L, c_vec=c_vec, n_vec=n_vec,
             b_vec=b_vec, chi_s=np.asarray(chi_s, dtype=np.float64))
    print(f"Saved → {outpath}")
    print(f"r           = {r:.6f}   (repr {r!r})")
    print(f"K_hat       = {K:.4f}   (repr {K!r})")
    print(f"L_hat       = {L:.6f}   (repr {L!r})")
    print(f"||c_vec||_1 = {np.sum(np.abs(c_vec))!r}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "verify_tests.npz"
    main(out)
