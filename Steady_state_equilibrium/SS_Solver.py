import numpy as np
import matplotlib.pyplot as plt
from Individual_level.Households import Household
from State_level.Aggregates import Aggregator
from State_level.Production import Firm
from State_level.PublicSector import Government
from main import reimport


class SteadyStateEquilibrium:
    r"""
    Orchestrates the calculation of the General Steady State Equilibrium
    Ref: algo:steady_state_solution

    Coordinates:
    1. Firm behavior (Factor Prices)
    2. Household behavior (Optimization Loop)
    3. Government Policy (Budget Constraint)
    4. Aggregation (Market Clearing)
    """

    def __init__(self, params: dict, vectors: dict):
        r"""
        :param params: Dictionary of scalar parameters (A, alpha, delta, tau_l, etc.)
                       Matches structure in Ref: tab:calib_param
        :param vectors: Dictionary of vector parameters (omega, rho, imigr_rate, etc.)
        """
        reimport()

        self._validate_init_inputs(params, vectors)

        self.params = params.copy()
        self.vectors = vectors

        # Unpack essential vectors for aggregation
        self.omega = vectors['omega']           # size S x T matrix of population by age and time
        self.rho = vectors['rho']               # size S x T matrix of death probabilities by age and time
        self.i_rate = vectors['i_rate']         # size S x 1 vector of net migration rates by age
        self.I_total = vectors['I_total']       # size 1 x T vector of total net migration by time
        self.g_n = params['g_n']                # g_n = N_t+1 / N_t - 1
        
        self.S = params['S']
        self.R = params['R']

        # Initialize Economic Agents
        self.household = Household(self.params, self.rho)

        self.firm = Firm(
            A=params['A'],
            alpha=params['alpha'],
            delta=params['delta'],
            tau_c=params['tau_c']
        )

        self.aggregator = Aggregator(
            omega=self.omega,
            g_n=self.g_n,
            imigr_rate=self.i_rate,
            imigr_total=self.I_total,
            E=params['E']
        )

        self.gov = Government(
            tau_l=params['tau_l'],
            tau_k=params['tau_k'],
            tau_c=params['tau_c'],
            retirement_age=params['R'],
            entry_age=params['E']
        )

    def solve(
            self,
            r_guess: float = .1,
            BQ_guess: float = .05,
            tax_guess: float = .1,
            policy_mode: str = 'fix_pension',
            tax_type: str = 'tau_l',
            xi: float = 0.2,
            tol: float = 1e-6,
            max_iter: int = 100,
            debug: bool = False
    ):
        r"""
        Executes the Outer Loop of the Steady State Algorithm.

        :param r_guess: Initial guess for interest rate
        :param BQ_guess: Initial guess for Aggregate Bequests
        :param tax_guess: Initial guess for the fiscal variable (either tax rate or pension amount X)
        :param policy_mode: 'fix_tax' (solve for X) or 'fix_pension' (solve for specific tax)
        :param tax_type: The specific tax to adjust if in 'fix_pension' mode ('tau_l', 'tau_k', 'tau_c')
        :param xi: Dampening factor for updates (0 < xi <= 1)
        :param tol: Convergence tolerance
        :param max_iter: Maximum iterations
        :param debug: If True, prints detailed market clearing info at the end

        :return: Dictionary containing all Steady State variables
        """
        self._validate_solve_inputs(r_guess, BQ_guess, tax_guess, policy_mode, tax_type, xi, max_iter)

        # Current guesses
        r = r_guess
        BQ = BQ_guess
        curr_policy_var = tax_guess

        # Initialize Tax System based on assumptions:
        # 1. Reset all taxes to 0 (assumption: specific tax covers gap, others are zero)
        self.gov.tau_l = 0.0
        self.gov.tau_k = 0.0
        self.gov.tau_c = 0.0
        self._sync_params()

        # 2. Set the active guess
        if policy_mode == 'fix_tax':
            # tax_guess is the fixed rate for the chosen type
            if tax_type == 'tau_l':
                self.gov.tau_l = tax_guess
            elif tax_type == 'tau_k':
                self.gov.tau_k = tax_guess
            elif tax_type == 'tau_c':
                self.gov.tau_c = tax_guess
            else:
                raise ValueError(f"Unknown tax_type {tax_type}")

            # Initial guess for X is 0 as requested
            #curr_policy_var = 0.0

        elif policy_mode == 'fix_pension':
            # tax_guess is the initial guess for the rate of 'tax_type'
            if tax_type == 'tau_l':
                self.gov.tau_l = tax_guess
            elif tax_type == 'tau_k':
                self.gov.tau_k = tax_guess
            elif tax_type == 'tau_c':
                self.gov.tau_c = tax_guess
            else:
                raise ValueError(f"Unknown tax_type {tax_type}")

            # curr_policy_var will track the tax rate

        self._sync_params()

        # Result container
        final_res = None

        target_replacement = self.params['replacement_rate']

        for i in range(max_iter):
            # 1. Factor Prices (FPF)
            w = self.firm.get_wage_from_r(r)

            # Determine Fiscal State variables
            if policy_mode == 'fix_tax':
                # X is the variable being solved for
                X_val = curr_policy_var
            else:
                # policy_mode == 'fix_pension'
                # X is fixed based on replacement rate
                X_val = target_replacement * w
                self._check_finite_positive(X_val, "X_val")

                # Apply current tax guess to the active tax type
                if tax_type == 'tau_l':
                    self.gov.tau_l = curr_policy_var
                elif tax_type == 'tau_k':
                    self.gov.tau_k = curr_policy_var
                elif tax_type == 'tau_c':
                    self.gov.tau_c = curr_policy_var
                assert 0 <= curr_policy_var, f"Current tax rate guess must be >= 0, got {curr_policy_var}"


            self._sync_params()

            # Construct X vector
            X_vec = np.zeros(self.S)
            X_vec[self.gov.R_idx:] = X_val

            # 2. Households (Inner Loop)
            hh_res = self.household.solve_steady_state(
                w=w,
                r=r,
                X_vec=X_vec,
                BQ_val=BQ,
                omega=self.omega,
            )

            if hh_res is None:
                print(f"Iter {i}: HH Solver failed.")
                break

            c_vec, n_vec, b_vec = hh_res
            assert b_vec.shape[0] == self.omega.shape[0] + 1, "b_vec must be length S+1 (includes terminal asset)"
            # Check positivity for economically active part
            self._check_finite_positive(c_vec[self.params['E']:], "c_vec[E:]", strict=True)

            # 3. Aggregation
            L_new = self.aggregator.get_aggregate_labor(n_vec)
            K_new = self.aggregator.get_aggregate_capital(b_vec)
            BQ_new = self.aggregator.get_aggregate_bequests(b_vec, self.rho, r)

            # Clamp K and L to small positive values to avoid crashes in Production
            # and allow the solver to recover from bad guesses (negative K implies low r -> clamping K high r -> savings up)
            if K_new <= 0:
                print(f"Warning: K_new is negative ({K_new}). Clamping to 1e-9.")
                K_new = 1e-9
            if L_new <= 0:
                print(f"Warning: L_new is negative ({L_new}). Clamping to 1e-9.")
                L_new = 1e-9
            if BQ_new <= 0:
                print(f"Warning: BQ_new is negative ({BQ_new}). Clamping to 1e-9.")
                BQ_new = 1e-9

            self._check_finite_positive(L_new, "L_new")
            self._check_finite_positive(K_new, "K_new")
            self._check_finite_positive(BQ_new, "BQ_new")

            # 4. Production
            Y_new = self.firm.get_output(K_new, L_new)
            self._check_finite_positive(Y_new, "Y_new")

            if policy_mode == 'fix_tax':
                # Update X based on total revenue collected
                # Pass full b_vec (size S+1)

                total_revenue = self.gov.get_total_tax_revenue(
                    self.omega, w, r, n_vec, b_vec, Y_new, L_new, K_new, self.firm.delta
                )
                policy_new = self.gov.get_pension_benefits(total_revenue, self.omega)
            else:
                # Update specific tax rate to fund X
                if tax_type == 'tau_l':
                    policy_new = self.gov.get_required_tau_l(X_val, self.omega, w, n_vec)
                elif tax_type == 'tau_k':
                    policy_new = self.gov.get_required_tau_k(X_val, self.omega, r, b_vec)
                elif tax_type == 'tau_c':
                    policy_new = self.gov.get_required_tau_c(X_val, self.omega, Y_new, w, L_new, K_new, self.firm.delta)

            self._check_finite(policy_new, "policy_new")

            # 6. Interest Rate Update
            r_new = self.firm.get_interest_rate(K_new, L_new)

            # 7. Convergence Check
            r_diff = r_new - r
            BQ_diff = BQ_new - BQ
            policy_diff = policy_new - curr_policy_var

            error = max(abs(r_diff), abs(BQ_diff), abs(policy_diff))

            # Prepare result dictionary
            C_agg = self.aggregator.get_aggregate_consumption(c_vec)

            if debug and i % 50 == 0:
                print(f"Iter {i}: Error={error:.6f}")
                print(f"  guesses: r={r:.4f} | BQ={BQ:.4f} | PolVar={curr_policy_var:.4f}")
                print(f"  implied: r_new={r_new:.4f} | BQ_new={BQ_new:.4f} | policy_new={policy_new:.4f}")
                print(f"  aggregates: Y={Y_new:.4f} | K={K_new:.4f} | L={L_new:.4f} | C={C_agg:.4f} | w={w:.4f}")
                print(f"  b_R={b_vec[self.R]:.4f}, b_S={b_vec[self.S]:.4f}, c_old={c_vec[self.S-1]:.4f}\n")
                print(25*"---")
                #break
                
            final_res = {
                'r': r_new,
                'w': w,
                'BQ': BQ_new,
                'X_val': X_val,
                'tau_l': self.gov.tau_l,
                'tau_k': self.gov.tau_k,
                'tau_c': self.gov.tau_c,
                'Y': Y_new,
                'K': K_new,
                'L': L_new,
                'C': C_agg,
                'c_vec': c_vec,
                'n_vec': n_vec,
                'b_vec': b_vec,
                'X_vec': X_vec
            }

            if error < tol:
                print(f"Converged in {i} iterations, Error={error:.6f}")
                print(f"  guesses: r={r:.4f} | BQ={BQ:.4f} | PolVar={curr_policy_var:.4f}")
                print(f"  implied: r_new={r_new:.4f} | BQ_new={BQ_new:.4f} | policy_new={policy_new:.4f}")
                print(f"  aggregates: Y={Y_new:.4f} | K={K_new:.4f} | L={L_new:.4f} | C={C_agg:.4f} | w={w:.4f}\n")
                print(f"  b_R={b_vec[self.R]:.4f}, b_S={b_vec[self.S]:.4f}, c_old={c_vec[self.S-1]:.4f}\n")
                
                break

            # 8. Relaxation
            r = xi * r_new + (1 - xi) * r
            BQ = xi * BQ_new + (1 - xi) * BQ
            curr_policy_var = xi * policy_new + (1 - xi) * curr_policy_var

        # Debug Output
        if debug and final_res:
            print("\n--- DEBUG: Market Clearing Conditions ---")
            self.check_goods_market_clearing(final_res)
            self.check_euler_errors(final_res)
            for par, val in final_res.items():
                if isinstance(val, float):
                    print(f'{par}: {val:.4f}')

            self.check_vecs(final_res)
            
            print("-----------------------------------------")

        if final_res and error < tol:
            return final_res

        print("Max iterations reached or solver failed.")
        return final_res if debug else None

    def check_goods_market_clearing(self, ss_dict: dict):
        r"""
        Validates the solution using the Goods Market Clearing condition.
        Ref: eq:clearing_output_stat
        """
        Y = ss_dict['Y']
        C = ss_dict['C']
        K = ss_dict['K']
        b_vec = ss_dict['b_vec']

        growth_term = np.exp(self.params['g_y']) * (1 + self.g_n)
        Inv = K * (growth_term - (1 - self.params['delta']))

        # b_vec is S+1. Migrants bring assets corresponding to their age.
        # b_vec aligned with age s is b_vec[s-1].
        # i_rate is aligned with age s (index s-1).
        # We need b_vec[:-1] to align with i_rate.
        mig_term = np.sum(self.i_rate * self.I_total * b_vec[:-1]) * np.exp(self.params['g_y'])

        Y_demand = C + Inv - mig_term
        diff = Y - Y_demand

        print(f"Goods Market: Supply Y = {Y:.4f}, Demand = {Y_demand:.4f}, Abs Diff = {abs(diff):.6e}")
        return diff
        
    def check_vecs(self, ss_dict: dict):
        r"""
        Checks the vectors in the steady state solution.
        """
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 3, figsize=[10, 2.5])

        c_vec = ss_dict['c_vec'][self.params['E']:]
        ax[0].plot(np.arange(self.params['E'], self.params['S']), c_vec)
        ax[0].set_title('Consumption')
        ax[0].set_xlabel('Age')
        ax[0].set_ylabel('Consumption')

        n_vec = ss_dict['n_vec'][self.params['E']:]
        ax[1].plot(np.arange(self.params['E'], self.params['S']), n_vec)
        ax[1].set_title('Labor')
        ax[1].set_xlabel('Age')
        ax[1].set_ylabel('Labor')

        b_vec = ss_dict['b_vec'][self.params['E']:-1]
        ax[2].plot(np.arange(self.params['E'], self.params['S']), b_vec)
        ax[2].set_title('Savings')
        ax[2].set_xlabel('Age')
        ax[2].set_ylabel('Savings')
        
        plt.show()

    def check_euler_errors(self, ss_dict: dict):
        r"""Validates the Habit-Formation Euler Equation."""
        from Individual_level.HabitUtility import compute_M_from_c

        c_vec = ss_dict['c_vec']
        r = ss_dict['r']
        sigma = self.params['sigma']
        beta = self.params['beta_s']
        g_y = self.params['g_y']
        tau_k = self.gov.tau_k
        E, S = self.params['E'], self.params['S']

        M_vec = compute_M_from_c(c_vec, self.rho, self.params)
        r_net = r * (1 - tau_k)

        lhs = M_vec[E:S-1]
        rhs = beta[E:S-1] * (1 - self.rho[E:S-1]) * (1 + r_net) * np.exp(-sigma * g_y) * M_vec[E+1:S]
        euler_errs = lhs - rhs

        max_err = np.max(np.abs(euler_errs))
        print(f"Habit Euler Equation: Max Abs Error = {max_err:.6e}")
        return euler_errs


    # =======================================================================================
    # ---------------------------- Helper Assertion Methods ---------------------------------
    # =======================================================================================

    def _validate_init_inputs(self, params: dict, vectors: dict):
        r"""Validates input parameters and vectors for initialization."""
        self._check_type(params, dict, "params")
        self._check_type(vectors, dict, "vectors")

        required_params = ('A', 'alpha', 'delta', 'tau_c', 'tau_l', 'tau_k', 'R', 'E',
                           'g_n', 'g_y', 'beta', 'sigma', 'replacement_rate', 'ltilde',
                           'b_ellip', 'upsilon', 'h')
        self._check_keys(params, required_params, "params")

        required_vectors = ('omega', 'rho', 'i_rate', 'I_total')
        self._check_keys(vectors, required_vectors, "vectors")

        self._check_vector_shape(vectors['omega'], "omega", (None,))
        self._check_vector_shape(vectors['rho'], "rho", (None,))
        self._check_vector_shape(vectors['i_rate'], "i_rate", (None,))

        assert vectors['omega'].shape == vectors['rho'].shape, \
            f"omega shape {vectors['omega'].shape} must match rho shape {vectors['rho'].shape}"
        assert vectors['i_rate'].shape == vectors['omega'].shape, \
            f"i_rate shape {vectors['i_rate'].shape} must match omega shape {vectors['omega'].shape}"

        assert np.all((vectors['rho'] >= 0.) & (vectors['rho'] <= 1.)), "rho must be in [0, 1]"
        assert params['A'] > 0, f"A must be > 0, got {params['A']}"
        assert 0 < params['alpha'] < 1, f"alpha must be in (0,1), got {params['alpha']}"
        assert 0 <= params['delta'] <= 1, f"delta must be in [0,1], got {params['delta']}"
        assert params['R'] > params['E'], f"R must be > E, got R={params['R']}, E={params['E']}"
        assert all(0 <= params[t] < 1 for t in ('tau_l', 'tau_k', 'tau_c')), "tax rates must be in [0,1)"
        assert 0 <= params['h'] < 1, f"h must be in [0,1), got {params['h']}"


    def _validate_solve_inputs(self, r_guess, BQ_guess, tax_guess, policy_mode, tax_type, xi, max_iter):
        r"""Validates inputs for the solve method."""
        self._check_finite(r_guess, "r_guess")
        self._check_finite(BQ_guess, "BQ_guess")
        self._check_finite(tax_guess, "tax_guess")

        assert policy_mode in ('fix_tax', 'fix_pension'), \
            f"policy_mode must be 'fix_tax' or 'fix_pension', got {policy_mode}"
        assert tax_type in ('tau_l', 'tau_k', 'tau_c'), \
            f"tax_type must be one of ('tau_l','tau_k','tau_c'), got {tax_type}"

        assert 0 < xi <= 1, f"xi must be in (0,1], got {xi}"
        assert isinstance(max_iter, int) and max_iter > 0, f"max_iter must be positive int, got {max_iter}"
        assert self.params.get('replacement_rate', 0) >= 0, "replacement_rate must be >= 0"

    def _check_type(self, obj, expected_type, name):
        assert isinstance(obj, expected_type), f"{name} must be a {expected_type.__name__}, got {type(obj)}"

    def _check_keys(self, dictionary, required_keys, name):
        missing = [k for k in required_keys if k not in dictionary]
        assert not missing, f"Missing required keys in {name}: {missing}"

    def _check_vector_shape(self, vector, name, expected_ndim=(None,)):
        assert isinstance(vector, np.ndarray), f"{name} must be a numpy array"
        if expected_ndim[0] is None:  # Check only 1D requirement usually
            assert vector.ndim == 1, f"{name} must be a 1D numpy array"
        else:
            assert vector.ndim == len(expected_ndim), f"{name} dimension mismatch"

    def _check_finite(self, value, name):
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value)), f"{name} must be finite"
        else:
            assert np.isfinite(value), f"{name} must be finite, got {value}"

    def _check_finite_positive(self, value, name, strict=False):
        self._check_finite(value, name)
        if strict:
            if isinstance(value, np.ndarray):
                assert np.all(value > 0), f"{name} must be strictly positive"
            else:
                assert value > 0, f"{name} must be strictly positive, got {value}"
        else:
            if isinstance(value, np.ndarray):
                assert np.all(value >= 0), f"{name} must be positive"
            else:
                assert value >= 0, f"{name} must be positive, got {value}"

    def _sync_params(self):
        """Syncs the Government tax rates to the main params dict used by Households"""
        self.params['tau_l'] = self.gov.tau_l
        self.params['tau_k'] = self.gov.tau_k
        self.params['tau_c'] = self.gov.tau_c

    # ======================================================================================