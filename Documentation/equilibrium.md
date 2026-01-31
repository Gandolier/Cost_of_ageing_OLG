# Equilibrium Section Analysis

## 1. Mathematical Derivation and Index Verification

The text presents shortened derivations for the Euler equations and the labor supply function. Below is the detailed reconstruction of these steps to verify the indexes and exponents.

### A. Inter-temporal Euler Equation (Equation 2.19)

**Goal:** Maximize utility with respect to savings $b_{s+1, t+1}$.

The maximization problem is:
$$
\max \sum_{k=E+1}^S \beta^{k-1} \left[ \prod_{u=E}^{k-1} (1-\rho_{u, t+k-E-1})\right] u(c_{k, t+k-E-1}, n_{k, t+k-E-1})
$$

We look at the trade-off between consumption at age $s$ (period $t$) and age $s+1$ (period $t+1$).
Let $\Omega_{s,t}$ be the cumulative survival probability up to age $s$ at time $t$.
Relevant terms in the sum:
$$
\Omega_{s,t} \beta^{s-1} u(c_{s,t}, \dots) + \Omega_{s+1,t+1} \beta^{s} u(c_{s+1,t+1}, \dots)
$$

**Budget Constraints:**
1. Period $t$: $c_{s,t} = \text{Income}_{s,t} - b_{s+1, t+1}$ (simplified)
2. Period $t+1$: $c_{s+1, t+1} = \text{Income}_{s+1,t+1} + (1+r_{t+1}(1-\tau^k_{s,t+1}))b_{s+1,t+1}$

**FOC w.r.t $b_{s+1, t+1}$:**
$$
\frac{\partial \mathcal{L}}{\partial b_{s+1,t+1}} = \Omega_{s,t} \beta^{s-1} \frac{\partial u}{\partial c_{s,t}} \frac{\partial c_{s,t}}{\partial b_{s+1,t+1}} + \Omega_{s+1,t+1} \beta^{s} \frac{\partial u}{\partial c_{s+1,t+1}} \frac{\partial c_{s+1,t+1}}{\partial b_{s+1,t+1}} = 0
$$

Derivatives from constraints:
$\frac{\partial c_{s,t}}{\partial b_{s+1,t+1}} = -1$
$\frac{\partial c_{s+1,t+1}}{\partial b_{s+1,t+1}} = (1+r_{t+1}(1-\tau^k_{s,t+1}))$

Relationship between survival probabilities:
$\Omega_{s+1,t+1} = \Omega_{s,t} \cdot (1-\rho_{s,t})$
*(Note: The text uses indices like $t+s-E-1$, but locally $t$ and $t+1$ suffice for the check).*

Substitute back:
$$
-\Omega_{s,t} \beta^{s-1} u'_{c_{s,t}} + \Omega_{s,t} (1-\rho_{s,t}) \beta^{s} u'_{c_{s+1,t+1}} (1+r_{t+1}(1-\tau^k_{s,t+1})) = 0
$$

Divide by $\Omega_{s,t} \beta^{s-1}$:
$$
u'_{c_{s,t}} = \beta (1-\rho_{s,t}) (1+r_{t+1}(1-\tau^k_{s,t+1})) u'_{c_{s+1,t+1}}
$$

**Marginal Utility of Consumption:**
$u(c) = \frac{c^{1-\sigma}}{1-\sigma} + \dots$
$u'(c) = c^{-\sigma}$

Substitute stationary variables ($c_{s,t} = \hat{c}_{s,t} e^{g_y t}$):
$$
(\hat{c}_{s,t} e^{g_y t})^{-\sigma} = \beta (1-\rho_{s,t}) (1+r_{t+1}(1-\tau^k_{s,t+1})) (\hat{c}_{s+1,t+1} e^{g_y (t+1)})^{-\sigma}
$$

Simplify exponents:
$e^{-\sigma g_y t} (\hat{c}_{s,t})^{-\sigma} = \beta (1-\rho_{s,t}) (1+r_{t+1}(1-\tau^k_{s,t+1})) e^{-\sigma g_y t} e^{-\sigma g_y} (\hat{c}_{s+1,t+1})^{-\sigma}$

Cancel $e^{-\sigma g_y t}$:
$$
(\hat{c}_{s,t})^{-\sigma} = e^{-\sigma g_y} \beta (1-\rho_{s,t}) (1+r_{t+1}(1-\tau^k_{s,t+1})) (\hat{c}_{s+1,t+1})^{-\sigma}
$$

**Conclusion on Eq 2.19:**
The text's equation (2.19) is:
$\Rightarrow (\hat{c}_{s,t})^{-\sigma} = e^{-\sigma g_y} \beta(1+r_{t+1}(1-\hat{\tau}^k_{s,t+1}))(\hat{c}_{s+1,t+1})^{-\sigma}$

**Error Found:** The term **$(1-\rho_{s,t})$** (survival probability) is missing from the final equation in the text. The summation in (2.16) explicitly includes survival probabilities, so they must appear in the Euler equation.

### B. Intra-temporal First Order Condition (Labor Supply)

**Goal:** Maximize utility with respect to labor $n_{s,t}$.

**FOC:**
$$
\frac{\partial u}{\partial n_{s,t}} + \lambda \frac{\partial (\text{budget})}{\partial n_{s,t}} = 0
$$
where $\lambda = \frac{\partial u}{\partial c_{s,t}} = (c_{s,t})^{-\sigma}$.

Budget constraint derivative:
$\frac{\partial \text{Income}}{\partial n_{s,t}} = (1-\tau^l_{s,t}) w_t$.

Marginal Dis-utility of Labor:
$u(\cdot, n) = \dots + B \left[ 1 - \left(\frac{n}{\tilde{l}}\right)^\upsilon \right]^{\frac{1}{\upsilon}}$ where $B = e^{t g_y(1-\sigma)} b$.

$$
\frac{\partial u}{\partial n} = B \cdot \frac{1}{\upsilon} \left[ 1 - \left(\frac{n}{\tilde{l}}\right)^\upsilon \right]^{\frac{1}{\upsilon}-1} \cdot \left( - \upsilon \left(\frac{n}{\tilde{l}}\right)^{\upsilon-1} \cdot \frac{1}{\tilde{l}} \right)
$$
$$
\frac{\partial u}{\partial n} = - B \frac{1}{\tilde{l}} \left(\frac{n}{\tilde{l}}\right)^{\upsilon-1} \left[ 1 - \left(\frac{n}{\tilde{l}}\right)^\upsilon \right]^{\frac{1-\upsilon}{\upsilon}}
$$

Equating marginal benefit of work (consumption) to marginal cost of work (disutility):
$$
(1-\tau^l_{s,t}) w_t (c_{s,t})^{-\sigma} = B \frac{1}{\tilde{l}} \left(\frac{n}{\tilde{l}}\right)^{\upsilon-1} \left[ 1 - \left(\frac{n}{\tilde{l}}\right)^\upsilon \right]^{\frac{1-\upsilon}{\upsilon}}
$$

Substitute stationary variables ($w_t = \hat{w}_t e^{g_y t}$, $c_{s,t} = \hat{c}_{s,t} e^{g_y t}$):
LHS: $(1-\tau^l_{s,t}) \hat{w}_t e^{g_y t} (\hat{c}_{s,t} e^{g_y t})^{-\sigma} = (1-\tau^l_{s,t}) \hat{w}_t (\hat{c}_{s,t})^{-\sigma} e^{g_y t (1-\sigma)}$

RHS: $e^{t g_y(1-\sigma)} b \frac{1}{\tilde{l}} \dots$

The term $e^{t g_y(1-\sigma)}$ cancels out on both sides.

**Conclusion on Eq 2.20 & 2.21:**
The text derivation is correct up to the exponent manipulation. However, there is a typo in the index of the tax rate in the final Equation (2.21). It uses $\hat{\tau}^L$ (uppercase L), but previous definitions used $\tau^l$ (lowercase l).

## 2. Algorithm Verification

The `Steady State solution algorithm` has a significant logical flow issue regarding the variables required to calculate $BQ$ (Bequests).

**The dependency chain:**
1. The algorithm calculates `BQ` **before** calculating savings `b_s`.
   > `State` $\bar{BQ} \gets \{ \bar{r}, \bar{b}_s, \bar{\omega}_s, \bar{\rho}_s, \tilde{g}_n \}$
   > `State` $\{ \bar{b_s} \}^S_{s=E+2} \gets \{ \dots, \bar{BQ}, \dots \}$
2. However, $BQ$ (bequests from the dead) depends on the savings of the population ($\bar{b}_s$).
3. In a steady state, the distribution of savings is constant. Therefore, you cannot calculate the aggregate bequest $BQ$ without knowing the savings distribution $\bar{b}_s$ first. But you cannot calculate the individual budget constraint (to find $\bar{b}_s$) without knowing the income from bequests ($BQ$).

**Missing Step/Input:**
The algorithm needs an **inner loop** or an additional guess.
Since $\bar{BQ}$ is a single scalar value representing aggregate bequests distributed to the living, it is most efficient to **guess $\bar{BQ}$** inside the rootfinding step, solve for household savings, calculate the resulting implied $BQ'$, and iterate until consistent.

Alternatively, because the algorithm describes a "Rootfinder" for $\bar{c}_{E+1}$ to satisfy the terminal condition $b_{S+1}=0$, this rootfinder actually implies solving the whole life-cycle. But the life-cycle budget constraint depends on $BQ$. The current text implies $BQ$ is calculated *before* the life-cycle savings, which is mathematically impossible as $BQ$ is a function of those savings.

## 3. Recommended Corrections for algo

\begin{equation}\label{eq:labour_function_explicit}
    n_{s,t} = \tilde{l} \left[ 1 + \left( \frac{b(\hat{c}_{s,t})^{\sigma}}{\tilde{l}\hat{w}_t (1-\hat{\tau}^l_{s,t})} \right)^{\frac{\upsilon}{\upsilon - 1}} \right]^{-\frac{1}{\upsilon}}
\end{equation}

Now we can define the algorithm of numerically finding the steady state of the system. For the algorithm we need at minimum three guesses of variables values for the outer loop, which will allow us to solve for the steady state values of the entire system. The choice of three variables for which we will make a guess is $r_t$, specific $\tau_{s,t}$ and $BQ_t$.

It is important to explain what is a \textit{specific} $\tau_{s,t}$. For the purposes of this research the three possible transition scenarios are created, they differ only in the way they deal with the fiscal tax burden due to pension payments. They correspond to financing the budget gap by three different taxes, all of which use just one tax rate. Depending on the scenario and a specific tax financing the transition we choose a tax rate variable for the outer loop solution.

In what follows, steady state values of variables are indicated by a bar sign.
\begin{algorithm}[H]
\caption{Steady State solution algorithm}
\begin{algorithmic}
    \Require make a guess for $\bar{r}$, a specific $\bar{\tau}$ and $\bar{BQ}$
    \Repeat

    \State $\bar{w} \gets \{\bar{r}, \bar{\tau^c}, A, \alpha, \delta \}$ according to \textcolor{Fuchsia}{\eqref{eq:wage_stat}} \\

    \hrulefill\\
        \Comment{A rootfinding algorithm}
        \State Make a guess for $\bar{c}_{E+1}$
        \State $\{ \bar{c_s} \}^S_{s=E+1} \gets \{ \bar{c_1}, \bar{r}, \bar{\tau}^k_s, \rho_s, \beta, \sigma, g_y \}$ according to \textcolor{Fuchsia}{\eqref{eq:euler_equations}}

        \State $\{ \bar{n_s} \}^S_{s=E+1} \gets \{ \bar{w}, \bar{c_s}, \bar{\tau}^l_s, \tilde{l}, \sigma, \upsilon, b \}$ according to \textcolor{Fuchsia}{\eqref{eq:labour_function_explicit}}

        \State $\{ \bar{b_s} \}^S_{s=E+2} \gets \{ \bar{c_s}, \bar{w}, \bar{n_s}, \bar{\tau}^l_s, \bar{r}, \bar{\tau}^k_s, \bar{X_s}, \bar{BQ}, g_y \}$ according to \textcolor{Fuchsia}{\eqref{eq:budget_constraint_stat}}, $\bar{b}_{s=E} = 0$

        \State Rootfinder optimizes initial $\bar{c}_{E+1}$ guess so as to satisfy $b_{s=S+1}=0$ -- in the last
        \State period of agent live savings are zero since he is aware of his inevitable death
        \State next year.\\
    \hrulefill\\

    \State Then, as above, find $\{ \bar{c_s} \}^S_{s=E+1} \rightarrow \{ \bar{n_s} \}^S_{s=E+1} \rightarrow \{ \bar{b_s} \}^S_{s=E+2}$

    \State $\bar{BQ}' \gets \{ \bar{r}, \bar{b}_s, \bar{\omega}_s, \bar{\rho}_s, \tilde{g}_n \}$ according to \textcolor{Fuchsia}{\eqref{eq:bequests_stat}}

    \State $\bar{r}' \gets \{ \bar{\tau}^c,   \bar{n}_s, \bar{b}_s, \bar{\omega_s}, \bar{I}, \bar{i}_s, \tilde{g}_n, \alpha, A, \delta \}$ according to \textcolor{Fuchsia}{\eqref{eq:key_rate_explicit}}

    \State $\bar{\tau}' \gets \{  \bar{n}_s, \bar{b}_s, \bar{w}, \bar{r}, \bar{X_s}, \bar{Y}, \bar{\omega}_s, \bar{N}_R, \delta \}$ according to \textcolor{Fuchsia}{\eqref{eq:tax_rates_stat}}\\

    \State Update initial guesses: $\{ \bar{r};\  \bar{\tau};\ \bar{BQ} \}_i = \{ \xi \bar{r}' + (1 - \xi)\bar{r};\ \xi \bar{\tau}' + (1 - \xi) \bar{\tau};\ \xi \bar{BQ}' + (1 - \xi) \bar{BQ} \} $

    \State \Until{$\{ (\bar{r}'- \bar{r})^2;\  (\bar{\tau}' - \bar{\tau})^2;\ (\bar{BQ}' - \bar{BQ})^2 \} < \{r_{tol};\ \tau_{tol};\ BQ_{tol} \}$}
\end{algorithmic}
\end{algorithm}

When the algorithm is converged you obtain all the variables for all ages of the model in the steady state, thus the algorithm allows to fully define the steady state equilibrium.

