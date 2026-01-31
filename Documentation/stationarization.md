# Stationarization Derivations

The model features two exogenous growth factors:
1.  **Technological progress**: $A_t = A e^{g_y t}$. This causes per-capita variables (like wages and savings) to grow at rate $e^{g_y t}$.
2.  **Population growth**: The population $N_t$ grows over time. Aggregate variables (like Total Capital and Total Labor) scale with the population size $\tilde{N}_t$.

To find a stationary equilibrium, we must transform variables to remove these trends.

## 1. Variable Transformations (Prerequisites)

Before deriving aggregate equations, we must define the stationary forms of the individual and demographic variables, as they are the building blocks for aggregates.

### 1.1 Individual Variables ($b, c, w$)
## Proof of Stationarity for Individual Variables ($w, c, b$)

We prove that along a balanced growth path, wages, consumption, and savings naturally grow at the rate of technological progress $g_y$. Therefore, dividing by $e^{g_y t}$ renders them constant (stationary).

### 1. Wage Growth (from Production)
The driving force is the labor-augmenting technological progress in the production function:
$$ Y_t = A K_t^{\alpha} (e^{g_y t} L_t)^{1-\alpha} $$

Let $L^{eff}_t = e^{g_y t} L_t$ be the effective labor. The production function can be written in intensive form per unit of effective labor. Let $k^{eff}_t = K_t / L^{eff}_t$.
$$ \frac{Y_t}{L^{eff}_t} = A \left( \frac{K_t}{L^{eff}_t} \right)^\alpha = A (k^{eff}_t)^\alpha $$

In a steady state, the capital-to-effective-labor ratio $k^{eff}$ is constant. Consequently, the output per effective unit of labor is constant.
The real wage $w_t$ is the marginal product of labor ($L_t$):
$$ w_t = \frac{\partial Y_t}{\partial L_t} = (1-\alpha) A K_t^{\alpha} (e^{g_y t} L_t)^{-\alpha} \cdot e^{g_y t} $$
$$ w_t = (1-\alpha) e^{g_y t} A \left( \frac{K_t}{e^{g_y t} L_t} \right)^\alpha $$
$$ w_t = e^{g_y t} \left[ (1-\alpha) A (k^{eff}_t)^\alpha \right] $$

Since the term in brackets is constant in steady state, the wage $w_t$ grows at exactly the rate $e^{g_y t}$.
$$ \hat{w}_t = \frac{w_t}{e^{g_y t}} = \text{const} $$

### 2. Consumption and Savings Growth (from Budget Constraint)
Consider the budget constraint for an individual of age $s$ at time $t$. If wages grow at $e^{g_y t}$, the income side of the equation scales by this factor. For the constraint to hold over time with a constant interest rate $r$ (which is constant because $MPK$ depends only on the ratio $k^{eff}$), consumption and savings must scale essentially by the same factor.

$$ c_{s,t} + b_{s+1, t+1} = (1-\tau) w_t n_{s,t} + (1+r) b_{s,t} + T_t $$

Substitute the trend forms $w_t = \hat{w} e^{g_y t}$:
$$ c_{s,t} + b_{s+1, t+1} = (1-\tau) (\hat{w} e^{g_y t}) n_{s,t} + (1+r) b_{s,t} + \dots $$

For a solution to exist where variables are stable relative to each other, $c$ and $b$ must share the same growth trend. Let's conjecture $c_{s,t} = \hat{c}_s e^{g_y t}$ and $b_{s,t} = \hat{b}_s e^{g_y t}$.

$$ \hat{c}_s e^{g_y t} + \hat{b}_{s+1} e^{g_y (t+1)} = (1-\tau) \hat{w} e^{g_y t} n_{s,t} + (1+r) \hat{b}_s e^{g_y t} $$

Divide the entire equation by $e^{g_y t}$:
$$ \hat{c}_s + \hat{b}_{s+1} e^{g_y} = (1-\tau) \hat{w} n_{s,t} + (1+r) \hat{b}_s $$

This results in a time-invariant (stationary) equation. This proves that defining $\hat{c} = c/e^{g_y t}$ and $\hat{b} = b/e^{g_y t}$ is the correct transformation.

### 3. Consistency with Preferences (Euler Equations)
Finally, we verify that these growth rates are consistent with the utility maximization first-order conditions.
$$ u(c, n) = \frac{c^{1-\sigma}}{1-\sigma} + e^{t g_y (1-\sigma)} v(n) $$

**Consumption Euler:**
$$ c_{s,t}^{-\sigma} = \beta (1+r) c_{s+1, t+1}^{-\sigma} $$
Substitute $c_{s,t} = \hat{c}_s e^{g_y t}$:
$$ (\hat{c}_s e^{g_y t})^{-\sigma} = \beta (1+r) (\hat{c}_{s+1} e^{g_y (t+1)})^{-\sigma} $$
$$ \hat{c}_s^{-\sigma} e^{-\sigma g_y t} = \beta (1+r) \hat{c}_{s+1}^{-\sigma} e^{-\sigma g_y t} e^{-\sigma g_y} $$
Cancel $e^{-\sigma g_y t}$:
$$ \hat{c}_s^{-\sigma} = \beta (1+r) e^{-\sigma g_y} \hat{c}_{s+1}^{-\sigma} $$
The time trend cancels out, confirming stationarity.

**Labor Supply FOC:**
$$ w_t (1-\tau) c_{s,t}^{-\sigma} = \frac{\partial \text{Disutility}}{\partial n} $$
LHS: $(\hat{w} e^{g_y t}) (1-\tau) (\hat{c}_s e^{g_y t})^{-\sigma} = \hat{w}(1-\tau)\hat{c}_s^{-\sigma} e^{g_y t (1-\sigma)}$
RHS: The explicit term in the utility function $e^{t g_y (1-\sigma)}$ scales the marginal disutility $v'(n)$.
$$ \hat{w}(1-\tau)\hat{c}_s^{-\sigma} e^{g_y t (1-\sigma)} = e^{t g_y (1-\sigma)} v'(n) $$
The term $e^{g_y t (1-\sigma)}$ cancels from both sides, leaving a stationary relationship for labor $n$.

**Conclusion:**
Dividing $w, c, b$ by $e^{g_y t}$ is the mathematically necessary transformation to remove the deterministic trend introduced by labor-augmenting technological progress.

### 1.2 Demographic Variables ($\omega, I, N$)
Aggregate variables are normalized by the effective working population $\tilde{N}_t = \sum_{s=E+1}^{S} \omega_{s,t}$.
Stationary population distribution:
$$ \hat{\omega}_{s,t} \equiv \frac{\omega_{s,t}}{\tilde{N}_t} $$
Stationary migration:
$$ \hat{I}_{t} \equiv \frac{I_{t}}{\tilde{N}_t} $$

**Crucial Relationship for Time-Shifting:**
Since capital is accumulated from the *previous* period, we often need to relate period $t-1$ aggregates to period $t$ deflators.
$$ 1 + \tilde{g}_{n,t} = \frac{\tilde{N}_t}{\tilde{N}_{t-1}} \implies \tilde{N}_{t-1} = \frac{\tilde{N}_t}{1 + \tilde{g}_{n,t}} $$

## 2. Aggregate Capital ($\hat{K}_t$)

We derive stationary capital $\hat{K}_t$ first, as it depends on the distribution of savings $\hat{b}$ and population $\hat{\omega}$.

**Original Equation:**
$$ K_t = \sum_{s=E+2}^{S} \left( \omega_{s-1,t-1} b_{s,t} + i_s I_{t-1} b_{s,t} \right) $$

**Derivation:**
1.  Divide both sides by the aggregate trend component $e^{g_y t} \tilde{N}_t$:
    $$ \hat{K}_t = \frac{K_t}{e^{g_y t} \tilde{N}_t} = \frac{1}{e^{g_y t} \tilde{N}_t} \sum_{s=E+2}^{S} b_{s,t} (\omega_{s-1,t-1} + i_s I_{t-1}) $$
2.  Substitute $b_{s,t} = \hat{b}_{s,t} e^{g_y t}$ (savings held at time $t$ are formed from income levels of time $t-1$ but valued at time $t$ prices/growth):
    $$ \hat{K}_t = \frac{1}{e^{g_y t} \tilde{N}_t} \sum_{s=E+2}^{S} \hat{b}_{s,t} e^{g_y t} (\omega_{s-1,t-1} + i_s I_{t-1}) $$
3.  Cancel $e^{g_y t}$:
    $$ \hat{K}_t = \frac{1}{\tilde{N}_t} \sum_{s=E+2}^{S} \hat{b}_{s,t} (\omega_{s-1,t-1} + i_s I_{t-1}) $$
4.  Use the population growth relationship to express $t-1$ variables in terms of $\tilde{N}_{t-1}$:
    $$ \hat{K}_t = \frac{1}{\tilde{N}_{t-1}(1+\tilde{g}_{n,t})} \sum_{s=E+2}^{S} \hat{b}_{s,t} (\omega_{s-1,t-1} + i_s I_{t-1}) $$
5.  Distribute $\tilde{N}_{t-1}$ into the sum to form stationary demographic variables $\hat{\omega} = \omega / \tilde{N}$ and $\hat{I} = I / \tilde{N}$:
    $$ \hat{K}_t = \frac{1}{1+\tilde{g}_{n,t}} \sum_{s=E+2}^{S} \hat{b}_{s,t} \left( \frac{\omega_{s-1,t-1}}{\tilde{N}_{t-1}} + i_s \frac{I_{t-1}}{\tilde{N}_{t-1}} \right) $$
6.  **Final Result**:
    $$ \hat{K}_t = \frac{1}{1+\tilde{g}_{n,t}} \sum_{s=E+2}^{S} \left( \hat{\omega}_{s-1,t-1} \hat{b}_{s,t} + i_s \hat{I}_{t-1} \hat{b}_{s,t} \right) $$

## 3. Aggregate Labor ($\hat{L}_t$)

**Original Equation:**
$$ L_t = \sum_{s=E+1}^{S} \omega_{s,t} n_{s,t} $$

**Derivation:**
1.  Divide by $\tilde{N}_t$. Note that $L_t$ is physical labor, not effective labor, so we only divide by population, not technology.
    $$ \hat{L}_t = \frac{L_t}{\tilde{N}_t} = \sum_{s=E+1}^{S} \frac{\omega_{s,t}}{\tilde{N}_t} n_{s,t} $$
2.  Substitute $\hat{\omega}_{s,t}$:
    $$ \hat{L}_t = \sum_{s=E+1}^{S} \hat{\omega}_{s,t} n_{s,t} $$

## 4. Production Function & Output ($\hat{Y}_t$)

**Original Equation:**
$$ Y_t = A K_t^{\alpha} (e^{g_y t} L_t)^{1-\alpha} $$

**Derivation:**
1.  Divide by $e^{g_y t} \tilde{N}_t$ to find $\hat{Y}_t$:
    $$ \hat{Y}_t = \frac{A K_t^{\alpha} (e^{g_y t} L_t)^{1-\alpha}}{e^{g_y t} \tilde{N}_t} $$
2.  The denominator can be split based on the exponents $\alpha + (1-\alpha) = 1$:
    $$ e^{g_y t} \tilde{N}_t = (e^{g_y t} \tilde{N}_t)^{\alpha} (e^{g_y t} \tilde{N}_t)^{1-\alpha} $$
3.  Group terms:
    $$ \hat{Y}_t = A \left( \frac{K_t}{e^{g_y t} \tilde{N}_t} \right)^{\alpha} \left( \frac{e^{g_y t} L_t}{e^{g_y t} \tilde{N}_t} \right)^{1-\alpha} $$
4.  Simplify. For the Labor term, $e^{g_y t}$ cancels out, leaving $L_t / \tilde{N}_t = \hat{L}_t$.
    $$ \hat{Y}_t = A (\hat{K}_t)^{\alpha} (\hat{L}_t)^{1-\alpha} $$

## 5. Interest Rate ($r_t$) and Wage ($\hat{w}_t$)

**Original Interest Rate Equation:**
$$ r_t = (1-\tau^c_t) \left[ \alpha \frac{Y_t}{K_t} - \delta \right] $$

**Derivation:**
1.  Since $Y_t$ and $K_t$ share the same trend factor $e^{g_y t} \tilde{N}_t$, their ratio is stationary:
    $$ \frac{Y_t}{K_t} = \frac{\hat{Y}_t e^{g_y t} \tilde{N}_t}{\hat{K}_t e^{g_y t} \tilde{N}_t} = \frac{\hat{Y}_t}{\hat{K}_t} $$
2.  From the stationary production function $\hat{Y}_t = A \hat{K}_t^{\alpha} \hat{L}_t^{1-\alpha}$:
    $$ \frac{\hat{Y}_t}{\hat{K}_t} = A \hat{K}_t^{\alpha-1} \hat{L}_t^{1-\alpha} = A \left( \frac{\hat{L}_t}{\hat{K}_t} \right)^{1-\alpha} $$
3.  Substitute the explicit sums for $\hat{L}_t$ and $\hat{K}_t$:
    $$ r_t = (1-\tau^c_t) \left[ \alpha A \left( \frac{\sum_{s=E+1}^{S} \hat{\omega}_{s,t} n_{s,t}}{\frac{1}{1+\tilde{g}_{n,t}} \sum_{s=E+2}^{S} (\hat{\omega}_{s-1,t-1} \hat{b}_{s,t} + i_s \hat{I}_{t-1} \hat{b}_{s,t})} \right)^{1-\alpha} - \delta \right] $$

*Note: In the original text, the summation for Capital in the denominator incorrectly ended at $E$ instead of $S$. This has been corrected.*

**Original Wage Equation:**
$$ w_t = (1-\alpha) \frac{Y_t}{L_t} $$

**Derivation:**
1.  Divide by $e^{g_y t}$:
    $$ \hat{w}_t = \frac{w_t}{e^{g_y t}} = (1-\alpha) \frac{Y_t}{L_t e^{g_y t}} $$
2.  Multiply top and bottom by $\tilde{N}_t$:
    $$ \hat{w}_t = (1-\alpha) \frac{Y_t}{e^{g_y t} \tilde{N}_t} \frac{\tilde{N}_t}{L_t} = (1-\alpha) \frac{\hat{Y}_t}{\hat{L}_t} $$

## 6. Budget Constraint

**Original Equation:**
$$ c_{s,t} + b_{s+1, t+1} = (1-\tau^l) w_t n_{s,t} + (1+r_t(1-\tau^k)) b_{s,t} + X_{s,t} + \frac{BQ_t}{\tilde{N}_t} $$

**Derivation:**
1.  Divide everything by $e^{g_y t}$:
    $$ \frac{c_{s,t}}{e^{g_y t}} + \frac{b_{s+1, t+1}}{e^{g_y t}} = (1-\tau^l) \frac{w_t}{e^{g_y t}} n_{s,t} + (1+r_t(1-\tau^k)) \frac{b_{s,t}}{e^{g_y t}} + \frac{X_{s,t}}{e^{g_y t}} + \frac{BQ_t}{e^{g_y t} \tilde{N}_t} $$
2.  The crucial step is handling the savings for the next period $b_{s+1, t+1}$. We must relate it to $\hat{b}_{s+1, t+1}$ which is deflated by $e^{g_y(t+1)}$.
    $$ \frac{b_{s+1, t+1}}{e^{g_y t}} = \frac{b_{s+1, t+1}}{e^{g_y(t+1)}} \frac{e^{g_y(t+1)}}{e^{g_y t}} = \hat{b}_{s+1, t+1} \cdot e^{g_y} $$
3.  **Final Result:**
    $$ \hat{c}_{s,t} + e^{g_y} \hat{b}_{s+1, t+1} = (1-\tau^l) \hat{w}_t n_{s,t} + (1+r_t(1-\tau^k)) \hat{b}_{s,t} + \hat{X}_{s,t} + \hat{BQ}_t $$

## 7. Goods Market Clearing

**Original Equation:**
$$ Y_t = C_t + Inv_t - \sum_{s=E+2}^{S} i_s I_t b_{s,t+1} $$

**Derivation:**
1.  Divide by $e^{g_y t} \tilde{N}_t$:
    $$ \hat{Y}_t = \hat{C}_t + \frac{Inv_t}{e^{g_y t} \tilde{N}_t} - \frac{1}{e^{g_y t} \tilde{N}_t} \sum i_s I_t b_{s,t+1} $$
2.  **Investment Term:** $Inv_t = K_{t+1} - (1-\delta) K_t$.
    $$ \frac{K_{t+1}}{e^{g_y t} \tilde{N}_t} = \frac{\hat{K}_{t+1} e^{g_y(t+1)} \tilde{N}_{t+1}}{e^{g_y t} \tilde{N}_t} = \hat{K}_{t+1} e^{g_y} (1+\tilde{g}_{n,t+1}) $$
    $$ \implies \hat{Inv}_t = e^{g_y}(1+\tilde{g}_{n,t+1})\hat{K}_{t+1} - (1-\delta)\hat{K}_t $$
3.  **Migration Term:**
    $$ \frac{1}{e^{g_y t} \tilde{N}_t} \sum i_s I_t b_{s,t+1} = \sum i_s \frac{I_t}{\tilde{N}_t} \frac{b_{s,t+1}}{e^{g_y t}} $$
    Using $b_{s,t+1} = \hat{b}_{s,t+1} e^{g_y(t+1)}$:
    $$ = \sum i_s \hat{I}_t \hat{b}_{s,t+1} \frac{e^{g_y(t+1)}}{e^{g_y t}} = e^{g_y} \sum i_s \hat{I}_t \hat{b}_{s,t+1} $$
4.  **Final Result:**
    $$ \hat{Y}_t = \hat{C}_t + \hat{Inv}_t - e^{g_y} \sum_{s=E+2}^{S} i_s \hat{I}_t \hat{b}_{s,t+1} $$

// ... existing code ...
## 8. Aggregate Bequests ($\hat{BQ}_t$)

**Original Equation:**
$$ BQ_t = (1+r_t) \sum^S_{s=E+2}\rho_{s-1, t-1}\omega_{s-1,t-1}b_{s,t} $$

**Derivation:**
1.  Divide by $e^{g_y t} \tilde{N}_t$:
    $$ \frac{BQ_t}{e^{g_y t} \tilde{N}_t} = \frac{1+r_t}{e^{g_y t} \tilde{N}_t} \sum^S_{s=E+2}\rho_{s-1, t-1}\omega_{s-1,t-1}b_{s,t} $$
2.  Substitute $b_{s,t} = \hat{b}_{s,t} e^{g_y t}$:
    $$ \hat{BQ}_t = \frac{1+r_t}{e^{g_y t} \tilde{N}_t} \sum^S_{s=E+2}\rho_{s-1, t-1}\omega_{s-1,t-1}\hat{b}_{s,t} e^{g_y t} $$
3.  Cancel $e^{g_y t}$ and use $\tilde{N}_t = \tilde{N}_{t-1}(1+\tilde{g}_{n,t})$:
    $$ \hat{BQ}_t = \frac{1+r_t}{\tilde{N}_{t-1}(1+\tilde{g}_{n,t})} \sum^S_{s=E+2}\rho_{s-1, t-1}\omega_{s-1,t-1}\hat{b}_{s,t} $$
4.  Distribute $\tilde{N}_{t-1}$ to form stationary population density $\hat{\omega}_{s-1,t-1} = \omega_{s-1,t-1}/\tilde{N}_{t-1}$:
    $$ \hat{BQ}_t = \frac{1+r_t}{1+\tilde{g}_{n,t}} \sum^S_{s=E+2}\rho_{s-1, t-1}\hat{\omega}_{s-1,t-1}\hat{b}_{s,t} $$

## 9. Pension System

### 9.1 Stationary Pension Payment ($\hat{X}_{s,t}$)
**Original Definition:**
$$ X_{s,t} = \frac{\text{Total Revenue}}{N_{R,t}} $$
$X_{s,t}$ is a per-capita transfer. Since other per-capita variables (wages, savings) grow at $e^{g_y t}$, pensions must also grow at this rate to remain relevant.
$$ \hat{X}_{s,t} \equiv \frac{X_{s,t}}{e^{g_y t}} $$

### 9.2 Stationary Tax Rates ($\tau$)
Tax rates are ratios (dimensionless). We must show they can be expressed purely in terms of stationary variables.

**Corporate Tax ($\tau^c_t$):**
$$ \tau^c_t = \frac{X_t N_{R,t}}{Y_t - w_t L_t - \delta K_t} $$
Divide numerator and denominator by $e^{g_y t} \tilde{N}_t$:
*   Numerator: $\frac{X_t N_{R,t}}{e^{g_y t} \tilde{N}_t} = \frac{X_t}{e^{g_y t}} \frac{N_{R,t}}{\tilde{N}_t} = \hat{X}_t \hat{N}_{R,t}$
*   Denominator: $\frac{Y_t - w_t L_t - \delta K_t}{e^{g_y t} \tilde{N}_t} = \hat{Y}_t - \frac{w_t L_t}{e^{g_y t} \tilde{N}_t} - \delta \hat{K}_t = \hat{Y}_t - \hat{w}_t \hat{L}_t - \delta \hat{K}_t$

$$ \hat{\tau}^c_t = \frac{\hat{X}_t \hat{N}_{R,t}}{\hat{Y}_t - \hat{w}_t \hat{L}_t - \delta \hat{K}_t} $$

**Labor Tax ($\tau^l_{s,t}$):**
$$ \tau^l_{s,t} = \frac{X_t N_{R,t}}{\sum \omega_{s,t} w_t n_{s,t}} $$
Divide by $e^{g_y t} \tilde{N}_t$:
*   Numerator becomes $\hat{X}_t \hat{N}_{R,t}$.
*   Denominator: $\frac{1}{e^{g_y t} \tilde{N}_t} \sum \omega_{s,t} w_t n_{s,t} = \sum \frac{\omega_{s,t}}{\tilde{N}_t} \frac{w_t}{e^{g_y t}} n_{s,t} = \sum \hat{\omega}_{s,t} \hat{w}_t n_{s,t}$

$$ \hat{\tau}^l_{s,t} = \frac{\hat{X}_t \hat{N}_{R,t}}{\hat{w}_t \sum_{s=E+1}^{R} \hat{\omega}_{s,t} n_{s,t}} $$

**Capital Tax ($\tau^k_{s,t}$):**
$$ \tau^k_{s,t} = \frac{X_t N_{R,t}}{\sum \omega_{s,t} r_t b_{s,t}} $$
Divide by $e^{g_y t} \tilde{N}_t$:
*   Numerator becomes $\hat{X}_t \hat{N}_{R,t}$.
*   Denominator: $\frac{1}{e^{g_y t} \tilde{N}_t} \sum \omega_{s,t} r_t b_{s,t} = r_t \sum \frac{\omega_{s,t}}{\tilde{N}_t} \frac{b_{s,t}}{e^{g_y t}} = r_t \sum \hat{\omega}_{s,t} \hat{b}_{s,t}$

$$ \hat{\tau}^k_{s,t} = \frac{\hat{X}_t \hat{N}_{R,t}}{r_t \sum_{s=E+1}^{S} \hat{\omega}_{s,t} \hat{b}_{s,t}} $$

## 10. Analysis of Wage Equation Stationarity
**Equation:**
$$ \hat{w}_t = A(1-\alpha) \left[ \frac{1}{A\alpha} \left( \frac{r_t}{1-\hat{\tau}^c_t} +\delta \right) \right]^{\frac{\alpha}{\alpha-1}} $$

**Validity Check:**
1.  **LHS:** $\hat{w}_t$ is stationary by definition.
2.  **RHS:**
    *   $A, \alpha, \delta$ are constant parameters.
    *   $r_t$ is the real interest rate. As derived in Section 5, $r_t$ depends on the ratio $\hat{Y}_t/\hat{K}_t$, which is constant in steady state. Thus, $r_t$ is stationary.
    *   $\hat{\tau}^c_t$ is the tax rate. As derived in Section 9.2, it is a ratio of stationary aggregates ($\hat{X}, \hat{N}_R, \hat{Y}$, etc.), making it stationary.
3.  **Conclusion:** The equation relates a stationary wage to a stationary interest rate and tax rate. It is dimensionally consistent and correctly stationarized. The presence of $\hat{\tau}^c_t$ is correct because the firm's first-order condition equates the *marginal product of capital* to the *user cost of capital*, which includes taxes.
    $$ MPK = \frac{r_t}{1-\tau^c_t} + \delta $$
    This determines the capital-labor ratio ($k^{eff}$), which uniquely determines the wage $\hat{w}_t$.


