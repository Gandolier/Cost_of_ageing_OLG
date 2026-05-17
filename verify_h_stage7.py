"""
Stage-7 verification: Turn habits on (h > 0) and compare against h=0 baseline.

Per spec: calibrate chi_s at h=0, then set h>0 and re-solve the SS using
the same chi_s. (Chi re-calibration is Stage 8.)

Pass criteria (from habits_coding.md):
  1. Consumption profile is slightly flatter than at h=0
  2. Equilibrium r has moved slightly downward (more savings due to habit smoothing)
  3. Root-finder converges (SS solver converges)

Usage:
    python verify_h_stage7.py [h_value]   (default: 0.1)
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


def build_data():
    """Load all data and build vectors + n_target."""
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
    return vectors, n_target, g_n


def main():
    h_habit = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1

    vectors, n_target, g_n = build_data()

    solver_params = {
        "r_guess": 0.05, "BQ_guess": 0.2, "tax_guess": .22,
        "policy_mode": "fix_tax", "tax_type": "tau_l",
        "xi": 0.1, "tol": 1e-3, "max_iter": 1000, "debug": True,
    }

    # ---- Step 1: Calibrate chi_s at h=0 ----
    print(f"\n{'='*60}")
    print(f"  Step 1: Calibrating chi_s at h=0")
    print(f"{'='*60}")
    params_h0 = {
        "A": 1.4, "alpha": .39, "delta": .05,
        "tau_c": 0., "tau_l": .22, "tau_k": 0.,
        "R": 62, "E": 20, "S": 100,
        "g_n": g_n, "g_y": .054, "beta": .905, "sigma": 1.97,
        "replacement_rate": .3, "ltilde": 1.,
        "b_ellip": .4309, "upsilon": 1.7648,
        "chi_s": np.ones(100), "h": 0.0,
    }
    chi_s, _ = calibrate_chi(
        params=params_h0, vectors=vectors, n_target=n_target,
        max_iter=50, tol=1e-3, xi_chi=0.99,
        ss_solve_kwargs=solver_params,
    )

    # ---- Step 2: Solve SS at h=0 (baseline) with calibrated chi_s ----
    print(f"\n{'='*60}")
    print(f"  Step 2: Final SS solve at h=0 (baseline)")
    print(f"{'='*60}")
    params_h0["chi_s"] = chi_s
    ss0 = SteadyStateEquilibrium(params_h0, vectors)
    res0 = ss0.solve(**solver_params)

    r0 = float(res0["r"])
    K0 = float(res0["K"])
    c0 = np.asarray(res0["c_vec"], dtype=np.float64)

    print(f"\n  h=0: r = {r0:.6f}, K = {K0:.4f}")
    print(f"  ||c_vec||_1 = {np.sum(np.abs(c0)):.6f}")
    print(f"  c_vec[20:] std = {np.std(c0[20:]):.6f}")
    print(f"  c_vec[20:] range = {np.max(c0[20:]) - np.min(c0[20:]):.6f}")

    # ---- Step 3: Solve SS at h>0 using SAME chi_s ----
    print(f"\n{'='*60}")
    print(f"  Step 3: SS solve at h={h_habit} (same chi_s from h=0)")
    print(f"{'='*60}")
    params_h = dict(params_h0)
    params_h["h"] = h_habit
    ss_h = SteadyStateEquilibrium(params_h, vectors)
    # Warm-start the h>0 household with the h=0 converged c_vec.
    # Without this, the cold-start (no-habit brent shoot) is too far
    # from the habit-aware solution for h >= 0.2.
    ss_h.household._c_vec_cache = c0[20:].copy()
    # Also warm-start the outer loop with h=0 equilibrium prices.
    solver_params_h = dict(solver_params)
    solver_params_h["r_guess"] = r0
    solver_params_h["BQ_guess"] = float(res0["BQ"])
    solver_params_h["tax_guess"] = float(res0.get("tau_l", 0.22))
    res_h = ss_h.solve(**solver_params_h)

    r_h = float(res_h["r"])
    K_h = float(res_h["K"])
    c_h = np.asarray(res_h["c_vec"], dtype=np.float64)

    print(f"\n  h={h_habit}: r = {r_h:.6f}, K = {K_h:.4f}")
    print(f"  ||c_vec||_1 = {np.sum(np.abs(c_h)):.6f}")
    print(f"  c_vec[20:] std = {np.std(c_h[20:]):.6f}")
    print(f"  c_vec[20:] range = {np.max(c_h[20:]) - np.min(c_h[20:]):.6f}")

    # ---- Compare ----
    print(f"\n{'='*60}")
    print(f"  Stage 7 Comparison: h=0 vs h={h_habit}")
    print(f"{'='*60}")

    E = 20
    std0 = np.std(c0[E:])
    stdh = np.std(c_h[E:])
    range0 = np.max(c0[E:]) - np.min(c0[E:])
    rangeh = np.max(c_h[E:]) - np.min(c_h[E:])

    print(f"\n  Baseline (h=0):  r = {r0:.6f},  K = {K0:.4f}")
    print(f"  Habit (h={h_habit}):  r = {r_h:.6f},  K = {K_h:.4f}")
    print(f"\n  c_vec[E:] std:   h=0 → {std0:.6f},  h={h_habit} → {stdh:.6f}")
    print(f"  c_vec[E:] range: h=0 → {range0:.6f},  h={h_habit} → {rangeh:.6f}")

    # Check 1: consumption flatter (use CV = std/mean, scale-independent)
    cv0 = std0 / np.mean(c0[E:])
    cvh = stdh / np.mean(c_h[E:])
    flatter = cvh < cv0
    print(f"\n  CHECK 1 — Consumption flatter (CV)? CV(h={h_habit}) < CV(h=0): "
          f"{cvh:.6f} < {cv0:.6f} → {'PASS' if flatter else 'FAIL'}")

    # Check 2: r moved down
    r_down = r_h < r0
    print(f"  CHECK 2 — r moved down? r(h={h_habit}) < r(h=0): "
          f"{r_h:.6f} < {r0:.6f} → {'PASS' if r_down else 'FAIL'}")

    # Check 3: root-finder convergence (SS solver converged with finite results)
    converged = np.isfinite(r_h) and np.isfinite(K_h)
    print(f"  CHECK 3 — Root-finder converged: → {'PASS' if converged else 'FAIL'}")

    overall = flatter and r_down and converged
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")

    # Save results
    np.savez("stage7_h0.npz", r=r0, K=K0, c_vec=c0,
             n_vec=np.asarray(res0["n_vec"], dtype=np.float64),
             b_vec=np.asarray(res0["b_vec"], dtype=np.float64),
             chi_s=np.asarray(chi_s, dtype=np.float64))
    np.savez(f"stage7_h{h_habit}.npz", r=r_h, K=K_h, c_vec=c_h,
             n_vec=np.asarray(res_h["n_vec"], dtype=np.float64),
             b_vec=np.asarray(res_h["b_vec"], dtype=np.float64),
             chi_s=np.asarray(chi_s, dtype=np.float64))
    print(f"\n  Saved stage7_h0.npz and stage7_h{h_habit}.npz")


if __name__ == "__main__":
    main()
