"""
Stage-2 unit tests for `Individual_level/HabitUtility.py`.

Pass criterion per Documentation/habits_coding.md §"Stage 2":
    At h = 0, compute_M_from_c(c_vec, rho, params) returns exactly
    c_vec ** (-sigma) for s in [E, S) and zero elsewhere.

Run:
    python test_habit_utility.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from Individual_level.HabitUtility import compute_delta_c, compute_M_from_c  # noqa: E402


def _base_params(h=0.0, E=20, S=100, sigma=1.97, beta=0.95, g_y=0.02):
    return {
        'h': h,
        'E': E,
        'S': S,
        'sigma': sigma,
        'beta': beta,
        'g_y': g_y,
    }


def _make_c_vec(E, S, rng):
    """A strictly positive, slowly growing lifecycle consumption profile,
    zero-padded for s < E (as in the rest of the codebase)."""
    c_vec = np.zeros(S)
    c_vec[E:] = 1.0 + 0.5 * rng.random(S - E) + 0.01 * np.arange(S - E)
    return c_vec


def test_h_zero_reduction():
    """At h=0: M == c_vec**(-sigma) on [E,S); zeros elsewhere."""
    rng = np.random.default_rng(0)
    params = _base_params(h=0.0)
    E, S, sigma = params['E'], params['S'], params['sigma']
    c_vec = _make_c_vec(E, S, rng)
    rho = rng.uniform(0.0, 0.1, size=S)

    M = compute_M_from_c(c_vec, rho, params)

    expected = np.zeros(S)
    expected[E:] = c_vec[E:] ** (-sigma)

    # Exact equality required at h=0 by the stage-2 spec.
    assert np.array_equal(M, expected), (
        f"M does not equal c**(-sigma) at h=0;"
        f" max|diff|={np.max(np.abs(M-expected)):.3e}"
    )
    # Belt-and-braces: zeros below E
    assert np.all(M[:E] == 0.0)


def test_delta_c_h_zero():
    """At h=0: delta_c == c_vec on [E,S), zeros below E."""
    rng = np.random.default_rng(1)
    params = _base_params(h=0.0)
    E, S = params['E'], params['S']
    c_vec = _make_c_vec(E, S, rng)

    delta_c = compute_delta_c(c_vec, params)

    assert np.all(delta_c[:E] == 0.0)
    assert np.array_equal(delta_c[E:], c_vec[E:])


def test_delta_c_boundary_at_E():
    """Boundary convention: delta_c[E] = c_vec[E] (no initial habit)."""
    rng = np.random.default_rng(2)
    params = _base_params(h=0.5)
    E, S = params['E'], params['S']
    c_vec = _make_c_vec(E, S, rng)

    delta_c = compute_delta_c(c_vec, params)

    # c_vec[E-1] is zero by construction, so delta_c[E] = c_vec[E]
    assert delta_c[E] == c_vec[E]


def test_delta_c_formula_positive_h():
    """For h>0: delta_c[s] = c[s] - h*exp(-g_y)*c[s-1] for s in (E, S)."""
    rng = np.random.default_rng(3)
    params = _base_params(h=0.5)
    E, S, h, g_y = params['E'], params['S'], params['h'], params['g_y']
    c_vec = _make_c_vec(E, S, rng)

    delta_c = compute_delta_c(c_vec, params)
    expected = np.zeros(S)
    expected[E] = c_vec[E]  # c[E-1] == 0
    expected[E+1:] = c_vec[E+1:] - h * np.exp(-g_y) * c_vec[E:S-1]

    np.testing.assert_allclose(delta_c, expected, rtol=0, atol=0)


def test_M_boundary_at_S_minus_1():
    """At the terminal age S-1: M[S-1] = delta_c[S-1]^(-sigma) (no future)."""
    rng = np.random.default_rng(4)
    params = _base_params(h=0.5)
    E, S, sigma = params['E'], params['S'], params['sigma']
    c_vec = _make_c_vec(E, S, rng)
    rho = rng.uniform(0.0, 0.1, size=S)

    delta_c = compute_delta_c(c_vec, params)
    M = compute_M_from_c(c_vec, rho, params)

    assert M[S-1] == delta_c[S-1] ** (-sigma)


def test_M_raises_on_nonpositive_delta_c():
    """Negative delta_c should raise ValueError (habit too strong)."""
    rng = np.random.default_rng(5)
    params = _base_params(h=0.99)  # very strong habit
    E, S = params['E'], params['S']
    # Strictly declining profile: c[s] = exp(-g_y) * 0.99 * c[s-1] is exactly on
    # the boundary; make it slightly worse to trip the check.
    c_vec = np.zeros(S)
    c_vec[E] = 1.0
    for s in range(E + 1, S):
        c_vec[s] = 0.5 * c_vec[s-1]
    rho = np.zeros(S)

    try:
        compute_M_from_c(c_vec, rho, params)
    except ValueError:
        return
    raise AssertionError("compute_M_from_c did not raise on non-positive delta_c")


def main():
    tests = [
        test_h_zero_reduction,
        test_delta_c_h_zero,
        test_delta_c_boundary_at_E,
        test_delta_c_formula_positive_h,
        test_M_boundary_at_S_minus_1,
        test_M_raises_on_nonpositive_delta_c,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} test(s) passed.")


if __name__ == "__main__":
    main()
