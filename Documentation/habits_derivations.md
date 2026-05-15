# Habit Formation in Consumption: Full Mathematical Derivation

This document re-derives, from scratch, every equation in the household side of the model that is affected by introducing internal habit formation in consumption preferences, while retaining the age-specific labor disutility scaler $\chi_s$ that was already added. The derivations mirror the style of Section 2 of the existing thesis chapter. Equations that are not affected by the habit modification — production, aggregation, government, market clearing, demographic dynamics, and the structural form of the budget constraint — are summarized briefly at the end with a justification for why they remain unchanged.

The boundary convention used throughout is that the agent enters economic life at age $E+1$ with no initial habit stock ($c_{E, t_0-1} \equiv 0$), no savings ($\hat b_{E+1} = 0$), and exits at age $S$ with no terminal savings ($\hat b_{S+1} = 0$). This is the same convention as the original paper.

---

## 1. The Modified Utility Function

The per-period utility, replacing equation (2.4) of the paper, is

$$u(c_{s,t}, c_{s-1,t-1}, n_{s,t}) \;=\; \frac{(c_{s,t} - h\, c_{s-1,t-1})^{1-\sigma}}{1-\sigma} \;+\; e^{tg_y(1-\sigma)}\,\chi_s\,b\left[1 - \left(\frac{n_{s,t}}{\tilde l}\right)^\upsilon\right]^{1/\upsilon}.$$

Two modifications relative to the paper's eq. (2.4):

1. The consumption argument is replaced by the *habit-adjusted consumption* $c_{s,t} - h\, c_{s-1, t-1}$, where $h \in [0, 1)$ is the habit intensity. The habit stock is the agent's *own* previous consumption (internal habits), not the aggregate consumption of others (which would be external habits).

2. The leisure term carries the age-specific multiplier $\chi_s$ as in the user's existing model.

The lifetime utility, replacing eq. (2.3), is

$$U(t_0) \;=\; \sum_{s=E+1}^{S}\beta^{s-1}\,\Pi_s\, u(c_{s}, c_{s-1}, n_s), \qquad \Pi_s = \prod_{u=E}^{s-1}(1-\rho_{u}),$$

where the time-index suppression follows the paper: $c_s$ is shorthand for $c_{s, t_0+s-E-1}$, the consumption of an agent aged $s$ when their economic life began in period $t_0$. The cumulative survival weight $\Pi_s$ is unchanged from the paper.

The habit boundary convention is that for $s = E+1$, the "previous consumption" $c_E$ enters with value zero. Mechanically this means $\Delta c_{E+1} \equiv c_{E+1} - h \cdot 0 = c_{E+1}$.

---

## 2. The Household Optimization Problem

The agent maximizes (2.3) subject to the per-period budget constraint

$$c_s + b_{s+1} \;=\; (1-\tau^l_{s,t})\,w_t\, n_s + \bigl(1 + r_t(1-\tau^k_{s,t})\bigr)\, b_s + X_s + BQ^{\text{receipt}}_{s,t}. \quad (\text{eq. 2.5, unchanged})$$

With the bequest-restriction modification made earlier, $BQ^{\text{receipt}}_{s,t}$ is zero for $s > R$ and equal to $BQ_t/\tilde N^W_t$ for $E+1 \leq s \leq R$, where $\tilde N^W_t$ is working-age population.

Form the Lagrangian with multipliers $\lambda_s$ on each period's budget constraint:

$$\mathcal{L} \;=\; \sum_{s=E+1}^{S}\beta^{s-1}\Pi_s\, u(c_s, c_{s-1}, n_s) \;-\; \sum_{s=E+1}^{S}\lambda_s \Bigl[c_s + b_{s+1} - (\text{period-$s$ income terms})\Bigr].$$

Three FOCs are derived: with respect to $c_s$, $b_{s+1}$, and $n_s$.

---

## 3. First-Order Conditions

### 3.1 FOC with respect to $c_s$

The decision variable $c_s$ enters the Lagrangian in three places:

(a) In the period-$s$ utility, as the consumption argument: $\frac{(c_s - hc_{s-1})^{1-\sigma}}{1-\sigma}$.

(b) In the period-$(s+1)$ utility, as the habit argument: $\frac{(c_{s+1} - hc_s)^{1-\sigma}}{1-\sigma}$. (Only for $s < S$; at $s = S$ there is no period $S+1$.)

(c) In the period-$s$ budget constraint, with coefficient $-1$: contributes $-\lambda_s$.

Compute the partial derivative.

**Term (a) derivative** (chain rule):
$$\frac{\partial}{\partial c_s}\left[\frac{(c_s - hc_{s-1})^{1-\sigma}}{1-\sigma}\right] = (c_s - hc_{s-1})^{-\sigma}.$$

**Term (b) derivative** (chain rule, noting the inner argument has $-h$ multiplying $c_s$):
$$\frac{\partial}{\partial c_s}\left[\frac{(c_{s+1} - hc_s)^{1-\sigma}}{1-\sigma}\right] = (c_{s+1} - hc_s)^{-\sigma} \cdot (-h).$$

This second contribution captures the agent's internalization of the fact that consuming more today raises tomorrow's habit, *lowering* tomorrow's marginal utility of (habit-adjusted) consumption. This is the defining feature of internal habits.

Setting $\partial \mathcal{L}/\partial c_s = 0$ for $E+1 \leq s \leq S-1$:

$$\beta^{s-1}\Pi_s\,(c_s - hc_{s-1})^{-\sigma} \;-\; h\,\beta^{s}\Pi_{s+1}\,(c_{s+1} - hc_s)^{-\sigma} \;=\; \lambda_s.$$

Using $\Pi_{s+1}/\Pi_s = 1-\rho_s$, factor out $\beta^{s-1}\Pi_s$:

$$\lambda_s \;=\; \beta^{s-1}\Pi_s\,\underbrace{\left[(c_s - hc_{s-1})^{-\sigma} \;-\; h\,\beta(1-\rho_s)\,(c_{s+1} - hc_s)^{-\sigma}\right]}_{\displaystyle \equiv M_s}. \quad (\star)$$

The bracketed quantity is the *effective marginal utility of consumption* — the net benefit of consuming an extra unit today, accounting for the future habit cost. Denote it $M_s$.

For $s = S$ (terminal age), the term (b) contribution does not exist (no period $S+1$ in the utility sum). The FOC reduces to:

$$\lambda_S = \beta^{S-1}\Pi_S\,(c_S - hc_{S-1})^{-\sigma}, \qquad M_S = (c_S - hc_{S-1})^{-\sigma}. \quad (\star')$$

**Limit check at $h = 0$.** Setting $h = 0$ in $(\star)$ gives $\lambda_s = \beta^{s-1}\Pi_s\, c_s^{-\sigma}$, which is the standard CRRA marginal utility times the survival-discount weight. This matches the no-habit benchmark. ✓

### 3.2 FOC with respect to $b_{s+1}$

The decision variable $b_{s+1}$ enters the Lagrangian in two places:

(a) In the period-$s$ budget constraint, with coefficient $-1$: contributes $-\lambda_s$.

(b) In the period-$(s+1)$ budget constraint, with coefficient $1 + r_{t+1}(1-\tau^k_{s+1, t+1})$: contributes $\lambda_{s+1}(1+r_{t+1}(1-\tau^k))$.

There is no utility derivative because $b_{s+1}$ does not enter the utility function. Setting the derivative to zero gives:

$$\lambda_s \;=\; \lambda_{s+1}\bigl(1 + r_{t+1}(1-\tau^k_{s+1})\bigr). \quad (\dagger)$$

This is identical in structure to the paper's no-habit case. The savings shadow-price compounds at the gross net-of-capital-tax rate, regardless of the consumption-side utility specification. The habit modification does not touch this equation.

### 3.3 FOC with respect to $n_s$

The decision variable $n_s$ enters in two places:

(a) The period-$s$ utility's leisure term: $e^{tg_y(1-\sigma)}\chi_s b\left[1 - (n_s/\tilde l)^\upsilon\right]^{1/\upsilon}$.

(b) The period-$s$ budget constraint, with coefficient $(1-\tau^l)w_t$ (wage income).

Compute the leisure-term derivative. Let $\eta = n_s/\tilde l$ for brevity. Then the leisure utility is $\chi_s b (1-\eta^\upsilon)^{1/\upsilon} \cdot e^{tg_y(1-\sigma)}$, and by the chain rule (twice):

$$\frac{d}{d\eta}\left[(1-\eta^\upsilon)^{1/\upsilon}\right] = \frac{1}{\upsilon}(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} \cdot (-\upsilon\,\eta^{\upsilon-1}) = -\eta^{\upsilon-1}(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon}.$$

Multiplying by $\frac{d\eta}{dn_s} = \frac{1}{\tilde l}$ gives

$$\frac{\partial u}{\partial n_s} = -\,e^{tg_y(1-\sigma)}\,\chi_s\,\frac{b}{\tilde l}\left(\frac{n_s}{\tilde l}\right)^{\upsilon-1}\left[1-\left(\frac{n_s}{\tilde l}\right)^\upsilon\right]^{(1-\upsilon)/\upsilon}.$$

Setting $\partial \mathcal{L}/\partial n_s = 0$:

$$-\,\beta^{s-1}\Pi_s\,e^{tg_y(1-\sigma)}\,\chi_s\,\frac{b}{\tilde l}\left(\frac{n_s}{\tilde l}\right)^{\upsilon-1}\left[1-\left(\frac{n_s}{\tilde l}\right)^\upsilon\right]^{(1-\upsilon)/\upsilon} \;+\; \lambda_s (1-\tau^l)w_t \;=\; 0.$$

Substituting $\lambda_s = \beta^{s-1}\Pi_s\, M_s$ from $(\star)$ and cancelling $\beta^{s-1}\Pi_s$ from both sides:

$$\boxed{\;e^{tg_y(1-\sigma)}\,\chi_s\,\frac{b}{\tilde l}\left(\frac{n_s}{\tilde l}\right)^{\upsilon-1}\left[1-\left(\frac{n_s}{\tilde l}\right)^\upsilon\right]^{(1-\upsilon)/\upsilon} \;=\; (1-\tau^l)\,w_t\, M_s\;} \quad (\ddagger)$$

The structural change compared to the paper's labor FOC: $c_s^{-\sigma}$ has been replaced by $M_s$. Since $M_s$ depends on both $c_s$ and $c_{s+1}$ (through $\Delta c_{s+1}$), the labor FOC at age $s$ now implicitly couples to consumption at age $s+1$, which it did not in the no-habit case.

**Limit check at $h = 0$.** $M_s = c_s^{-\sigma}$, so $(\ddagger)$ becomes $\chi_s (b/\tilde l)(\eta_s)^{\upsilon-1}(1-\eta_s^\upsilon)^{(1-\upsilon)/\upsilon}e^{tg_y(1-\sigma)} = (1-\tau^l) w_t c_s^{-\sigma}$, the existing labor FOC. ✓

---

## 4. The Euler Equation

Combine $(\star)$ for $\lambda_s$ and $(\dagger)$ for the savings condition:

$$\beta^{s-1}\Pi_s\, M_s \;=\; \bigl(1+r_{t+1}(1-\tau^k)\bigr)\, \beta^{s}\Pi_{s+1}\, M_{s+1}.$$

Divide both sides by $\beta^{s-1}\Pi_s$ and apply $\Pi_{s+1}/\Pi_s = 1-\rho_s$:

$$\boxed{\;M_s \;=\; \beta(1-\rho_s)\bigl(1+r_{t+1}(1-\tau^k)\bigr)\, M_{s+1}.\;} \quad (\text{Euler in } M)$$

This is the new Euler equation, replacing equation (2.26) of the paper.

In the variable $M_s$, the structure is *two-period coupled* — knowing $M_s$ pins down $M_{s+1}$ via the Euler. This looks deceptively similar to the no-habit Euler. The complexity is hidden inside the definition of $M_s$: it is a function of two consecutive consumption levels ($c_s$ and $c_{s+1}$), so the Euler in $M_s$ becomes a *three-period coupled* relation when expanded in consumption levels.

To see this expansion, substitute the definition of $M_s$ and $M_{s+1}$ from $(\star)$:

$$\underbrace{(c_s - hc_{s-1})^{-\sigma} - h\beta(1-\rho_s)(c_{s+1} - hc_s)^{-\sigma}}_{M_s} \;=\; \beta(1-\rho_s)(1+r_{t+1}(1-\tau^k))\;\underbrace{\bigl[(c_{s+1} - hc_s)^{-\sigma} - h\beta(1-\rho_{s+1})(c_{s+2} - hc_{s+1})^{-\sigma}\bigr]}_{M_{s+1}}.$$

This equation involves four consumption levels: $c_{s-1}, c_s, c_{s+1}, c_{s+2}$. The implication for solution methods is significant — forward shooting from a single initial condition $c_{E+1}$ is no longer well-posed.

**Limit check at $h = 0$.** $M_s = c_s^{-\sigma}$, so the Euler in $M$ becomes $c_s^{-\sigma} = \beta(1-\rho_s)(1+r(1-\tau^k)) c_{s+1}^{-\sigma}$, which is the paper's eq. (2.26). ✓

---

## 5. Stationarization

The stationarization procedure follows the paper's Section 2.5. Define stationary consumption $c_{s,t} = e^{tg_y}\hat c_{s,t}$ where, in steady state, $\hat c_{s,t}$ depends only on age $s$. Analogous transformations apply to all other variables (Table 1 of the paper), in particular $w_t = e^{tg_y}\hat w_t$ and $b_{s,t} = e^{tg_y}\hat b_{s,t}$.

### 5.1 Stationary Habit-Adjusted Consumption

The variable $\Delta c_{s,t} \equiv c_{s,t} - h c_{s-1,t-1}$ contains consumption at two adjacent calendar times. Substituting:

$$\Delta c_{s,t} \;=\; e^{tg_y}\hat c_s - h\, e^{(t-1)g_y}\hat c_{s-1} \;=\; e^{tg_y}\left[\hat c_s - h\,e^{-g_y}\hat c_{s-1}\right].$$

Define the *stationary habit-adjusted consumption*:

$$\boxed{\;\widehat{\Delta c}_s \;\equiv\; \hat c_s - h\,e^{-g_y}\,\hat c_{s-1}\;}$$

Then $\Delta c_{s,t} = e^{tg_y}\widehat{\Delta c}_s$ and $\Delta c_{s,t}^{-\sigma} = e^{-\sigma tg_y}\widehat{\Delta c}_s^{-\sigma}$.

The boundary condition $c_E \equiv 0$ propagates as $\hat c_E = 0$, so $\widehat{\Delta c}_{E+1} = \hat c_{E+1}$.

The factor $e^{-g_y}$ appearing in the definition of $\widehat{\Delta c}_s$ reflects the fact that one period ago, in calendar time, the productivity trend was a factor of $e^{g_y}$ smaller — so a unit of last period's consumption corresponds to $e^{-g_y}$ units in current detrended terms. This is necessary for $\widehat{\Delta c}_s$ to itself be a stationary quantity along the balanced growth path.

### 5.2 Stationary Effective Marginal Utility

Substitute into $M_s$:

$$M_s = e^{-\sigma tg_y}\widehat{\Delta c}_s^{-\sigma} - h\beta(1-\rho_s)\, e^{-\sigma(t+1)g_y}\widehat{\Delta c}_{s+1}^{-\sigma}.$$

Factor out $e^{-\sigma tg_y}$:

$$M_s = e^{-\sigma tg_y}\underbrace{\left[\widehat{\Delta c}_s^{-\sigma} - h\beta(1-\rho_s)\,e^{-\sigma g_y}\widehat{\Delta c}_{s+1}^{-\sigma}\right]}_{\displaystyle \equiv \hat M_s}.$$

Therefore:

$$\boxed{\;\hat M_s \;\equiv\; \widehat{\Delta c}_s^{-\sigma} - h\,\beta(1-\rho_s)\,e^{-\sigma g_y}\,\widehat{\Delta c}_{s+1}^{-\sigma}\;}$$

with boundary $\hat M_S = \widehat{\Delta c}_S^{-\sigma}$ (no future contribution, inherited from the corresponding boundary of $M_S$).

### 5.3 Stationary Euler

Substitute into the non-stationary Euler:

$$e^{-\sigma tg_y}\hat M_s \;=\; \beta(1-\rho_s)\bigl(1+r_{t+1}(1-\tau^k)\bigr)\,e^{-\sigma(t+1)g_y}\hat M_{s+1}.$$

In steady state, $r_{t+1} = r$ is constant. Dividing both sides by $e^{-\sigma tg_y}$:

$$\boxed{\;\hat M_s \;=\; \beta(1-\rho_s)\bigl(1+r(1-\tau^k)\bigr)\, e^{-\sigma g_y}\,\hat M_{s+1}\;} \quad (\text{stationary Euler})$$

This replaces equation (2.26) of the paper.

**Limit check at $h = 0$.** $\hat M_s = \hat c_s^{-\sigma}$, and the stationary Euler reduces to the paper's $\hat c_s^{-\sigma} = \beta(1-\rho_s)(1+r(1-\tau^k))e^{-\sigma g_y}\hat c_{s+1}^{-\sigma}$. ✓

**Expanded three-period form.** Substituting $\hat M_s$ and $\hat M_{s+1}$:

$$\widehat{\Delta c}_s^{-\sigma} - h\beta(1-\rho_s)\,e^{-\sigma g_y}\widehat{\Delta c}_{s+1}^{-\sigma} \;=\; \beta(1-\rho_s)(1+r(1-\tau^k))\,e^{-\sigma g_y}\left[\widehat{\Delta c}_{s+1}^{-\sigma} - h\beta(1-\rho_{s+1})\,e^{-\sigma g_y}\widehat{\Delta c}_{s+2}^{-\sigma}\right].$$

This is the fully expanded stationary Euler, involving four consecutive habit-adjusted consumptions ($\widehat{\Delta c}_s, \widehat{\Delta c}_{s+1}, \widehat{\Delta c}_{s+2}$) and equivalently four consecutive consumption levels through $\widehat{\Delta c}_s = \hat c_s - he^{-g_y}\hat c_{s-1}$.

### 5.4 Stationary Labor FOC

Substitute the stationary transformations into $(\ddagger)$. The left side already carries the factor $e^{tg_y(1-\sigma)}$ explicitly. The right side: $w_t M_s = e^{tg_y}\hat w \cdot e^{-\sigma tg_y}\hat M_s = e^{tg_y(1-\sigma)}\hat w\, \hat M_s$. The $e^{tg_y(1-\sigma)}$ factor cancels uniformly, leaving the stationary labor FOC:

$$\boxed{\;\chi_s\,\frac{b}{\tilde l}\left(\frac{n_s}{\tilde l}\right)^{\upsilon-1}\left[1-\left(\frac{n_s}{\tilde l}\right)^\upsilon\right]^{(1-\upsilon)/\upsilon} \;=\; (1-\tau^l)\,\hat w\,\hat M_s\;}$$

The right-hand side has changed from the original paper's $(1-\tau^l)\hat w \hat c_s^{-\sigma}$ to $(1-\tau^l)\hat w \hat M_s$. At $h = 0$, $\hat M_s = \hat c_s^{-\sigma}$ and the paper's form is recovered.

### 5.5 Closed-Form Solution for $n_s$

Apply the algebraic identity from the paper (Appendix, used implicitly in the derivation of eq. 2.28). Let $\eta = n_s / \tilde l$. Then:

$$\eta^{\upsilon-1}(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} = (\eta^{-\upsilon} - 1)^{(1-\upsilon)/\upsilon}.$$

*Derivation of the identity:* factor $(1-\eta^\upsilon) = \eta^\upsilon (\eta^{-\upsilon} - 1)$, so $(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} = \eta^{\upsilon \cdot (1-\upsilon)/\upsilon}(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} = \eta^{1-\upsilon}(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}$. Multiplying by $\eta^{\upsilon-1}$ gives $\eta^0 = 1$ times $(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}$, as claimed.

Apply to the labor FOC:

$$\chi_s\,\frac{b}{\tilde l}\,(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} \;=\; (1-\tau^l)\,\hat w\,\hat M_s.$$

Isolate $(\eta^{-\upsilon}-1)$:

$$(\eta^{-\upsilon} - 1)^{(1-\upsilon)/\upsilon} \;=\; \frac{(1-\tau^l)\,\hat w\,\hat M_s\,\tilde l}{\chi_s\, b}.$$

Raise both sides to the power $\upsilon/(1-\upsilon)$. Note that since $\upsilon > 1$, we have $1-\upsilon < 0$, so $\upsilon/(1-\upsilon) = -\upsilon/(\upsilon-1) < 0$. Equivalently, we can flip the fraction inside and use the positive exponent $\upsilon/(\upsilon-1)$:

$$\eta^{-\upsilon} - 1 \;=\; \left[\frac{\chi_s\, b}{(1-\tau^l)\,\hat w\,\hat M_s\,\tilde l}\right]^{\upsilon/(\upsilon-1)}.$$

Solve for $\eta$:

$$\eta^{-\upsilon} \;=\; 1 + \left[\frac{\chi_s\, b}{\tilde l\,\hat w\,(1-\tau^l)\,\hat M_s}\right]^{\upsilon/(\upsilon-1)}.$$

Finally, with $n_s = \tilde l\, \eta$:

$$\boxed{\;n_s \;=\; \tilde l\,\left[1 + \left(\frac{\chi_s\, b}{\tilde l\,\hat w\,(1-\tau^l)\,\hat M_s}\right)^{\upsilon/(\upsilon-1)}\right]^{-1/\upsilon}\;} \quad (\text{stationary } n_s)$$

This replaces equation (2.28) of the paper.

**Limit check at $h = 0$.** $\hat M_s = \hat c_s^{-\sigma}$, so the term inside parentheses becomes $\chi_s b \hat c_s^{\sigma} / (\tilde l \hat w (1-\tau^l))$. The formula reduces to the existing labor supply equation. ✓

**Limit check at $\chi_s \to \infty$.** The bracket inside parentheses diverges; raised to the positive power $\upsilon/(\upsilon-1)$ it diverges; the outer bracket grows without bound; raised to $-1/\upsilon < 0$ goes to zero; $n_s \to 0$. Old agents with high disutility retire. ✓

**Limit check at $\hat M_s \to \infty$** (very low consumption → infinite marginal utility). The bracket inside parentheses goes to zero, the outer bracket goes to 1, $n_s \to \tilde l$. Agents in extreme deprivation work the entire time endowment to escape it. ✓

---

## 6. The Stationary Budget Constraint and Boundary Conditions

The budget constraint contains no utility terms, so habit formation does not affect it. The stationarization in Section 2.5 of the paper applies unchanged. Equation (2.25) of the paper, in the form used in your code:

$$\hat c_s \;=\; (1-\tau^l)\,\hat w\, n_s + \bigl(1+r(1-\tau^k)\bigr)\,\hat b_s + \hat X_s + \widehat{BQ}_s^{\text{receipt}} - e^{g_y}\hat b_{s+1}.$$

With the working-age bequest distribution rule from the user's earlier modification:
- $\widehat{BQ}_s^{\text{receipt}} = \widehat{BQ}/\sum_{u=E+1}^{R}\hat\omega_u$ for $E+1 \leq s \leq R$,
- $\widehat{BQ}_s^{\text{receipt}} = 0$ for $R < s \leq S$.

The boundary conditions are:
- $\hat b_{E+1} = 0$ (no wealth at entry),
- $\hat b_{S+1} = 0$ (no wealth at death),
- $\hat c_E = 0$ (no initial habit).

These three boundary conditions are the same as in the no-habit model. The habit modification adds no new boundary conditions and does not modify the existing ones.

---

## 7. Modified $\chi_s$ Calibration

The labor profile calibration inverts the labor FOC at a target $\bar n_s$ profile to solve for $\chi_s$. From the stationary labor FOC:

$$\chi_s\,\frac{b}{\tilde l}\,(\eta_s^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} \;=\; (1-\tau^l)\,\hat w\,\hat M_s,$$

with $\eta_s = \bar n_s/\tilde l$ (the target labor share). Solving for $\chi_s$:

$$\chi_s = \frac{(1-\tau^l)\,\hat w\,\hat M_s\,\tilde l}{b\,(\eta_s^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}}.$$

Moving the $(\eta_s^{-\upsilon}-1)$ factor to the numerator by inverting its exponent's sign:

$$\boxed{\;\chi_s \;=\; \frac{\tilde l\,(1-\tau^l)\,\hat w\,\hat M_s}{b}\,\left[\left(\frac{\bar n_s}{\tilde l}\right)^{-\upsilon} - 1\right]^{(\upsilon-1)/\upsilon}\;} \quad (\text{modified } \chi_s)$$

**Limit check at $h = 0$.** $\hat M_s = \hat c_s^{-\sigma}$, and the formula reduces to

$$\chi_s = \frac{\tilde l (1-\tau^l)\hat w \hat c_s^{-\sigma}}{b}\,[\cdot]^{(\upsilon-1)/\upsilon} = \frac{\tilde l (1-\tau^l)\hat w}{b\, \hat c_s^\sigma}\,[\cdot]^{(\upsilon-1)/\upsilon},$$

which matches the existing $\chi_s$ closed-form inverse in the user's `update_chi_from_foc` function. ✓

The calibration loop architecture is unchanged: starting from $\chi_s^{(0)} = 1$, alternate between (a) solving the full SS for given $\chi_s$ to obtain $\hat c$, $\hat w$, $\tau^l$, and (b) updating $\chi_s$ via the closed-form formula above using the equilibrium quantities. The only change is that $\hat M_s$ is computed from $\hat c$ (and $\rho$, $h$, $\beta$, $\sigma$, $g_y$ via Section 5.2) before applying the formula.

---

## 8. What Does Not Change

The following components of the model are unaffected by the habit modification and retain their paper-version equations. The reason in each case is that they do not involve the household utility function.

### 8.1 Production Function and Firm FOCs

The firm's problem (eqs. 2.6, 2.7) depends only on aggregate capital, aggregate effective labor, and technology parameters $A, \alpha, \delta, g_y, \tau^c$. The wage and interest rate equations (eqs. 2.8, 2.9, 2.10) are derived from the firm's profit maximization and have no dependence on household utility. Their stationary forms (eqs. 2.21, 2.22, 2.23) are unchanged.

### 8.2 Aggregator

Aggregate labor (eq. 2.17), aggregate capital (eq. 2.18), aggregate consumption (the definition in eq. 2.20), and aggregate bequests (eq. 2.19) are linear functions of individual decisions multiplied by population weights. Habit formation modifies the *individual decisions* $\hat c_s, n_s, \hat b_s$, which feed into the aggregators with their new values, but the aggregator formulas themselves are unchanged.

### 8.3 Government

The pension distribution rule and the tax revenue / required tax rate equations (eqs. 2.15, 2.16, 2.24) involve aggregate labor income, aggregate capital income, and aggregate profit — all of which are computed via the (unchanged) aggregator. The government equations are therefore structurally unchanged.

### 8.4 Market Clearing

The goods market clearing (eq. 2.13, stationary form 2.20), labor market (eq. 2.11, stationary 2.17), and capital market (eq. 2.12, stationary 2.18) are accounting identities. Habit formation does not introduce or eliminate any economic flows; it only redistributes them across ages. Market clearing therefore continues to hold by construction whenever the individual-level decisions satisfy the new FOCs.

### 8.5 Demography

The population dynamics (eq. 2.1, 2.2) depend only on fertility, mortality, and migration — exogenous to the household problem. They are unchanged.

### 8.6 Stationarization of Aggregate Variables

The Table 1 stationarization conventions for $\hat\omega, \hat L, \hat K, \hat Y, \hat C, \widehat{BQ}, \hat I, \hat X, \hat N_R$, etc., are based on the trend behavior of aggregates along the balanced growth path. These trend behaviors (output growing at $g_y + g_n$ in absolute terms, etc.) are properties of the production technology and demographic dynamics, not of the household utility specification. The stationarization rules in Table 1 therefore remain valid under the habit modification.

### 8.7 The Budget Constraint Structurally

The per-period budget constraint (eq. 2.5) does not depend on utility specification. Its stationary form (eq. 2.25) is unchanged. The only subtle point is that with the bequest-restriction modification, $BQ_t/\tilde N_t$ becomes $BQ_t^{\text{receipt}}_s$ which is age-specific, but that was already the case before introducing habits.

---

## 9. The Modified Equilibrium Algorithm

The non-trivial consequence of the three-period-coupled Euler is that the existing forward shooting method in `Individual_level/Households.py` is no longer valid.

### 9.1 Why Forward Shooting Fails

In the no-habit model, the Euler equation $\hat c_s^{-\sigma} = \beta(1-\rho_s)(1+r_{\text{net}})e^{-\sigma g_y}\hat c_{s+1}^{-\sigma}$ relates two consecutive consumption levels. Given $\hat c_s$, this uniquely determines $\hat c_{s+1}$:

$$\hat c_{s+1} = \hat c_s \cdot \left[\beta(1-\rho_s)(1+r_{\text{net}})\right]^{1/\sigma} e^{-g_y}.$$

Therefore, a single initial guess $\hat c_{E+1}$ determines the entire consumption profile $\hat c_{E+1}, \hat c_{E+2}, \ldots, \hat c_S$ by forward iteration. The initial guess is then adjusted to satisfy the one terminal condition $\hat b_{S+1} = 0$. This is the scalar Brent's-method shooting that the existing code implements.

In the habit model, the Euler relates four consecutive consumption levels ($\hat c_{s-1}, \hat c_s, \hat c_{s+1}, \hat c_{s+2}$). Given $\hat c_{s-1}$ and $\hat c_s$, the Euler is one equation in two unknowns ($\hat c_{s+1}$ and $\hat c_{s+2}$) and cannot be solved forward.

### 9.2 The Global Root-Finder

Treat the entire consumption profile $\hat c_{E+1}, \hat c_{E+2}, \ldots, \hat c_S$ as a vector of $S - E$ unknowns. The constraints are:

(a) $S - E - 1$ stationary Euler residuals at $s = E+1, \ldots, S-1$:
$$\hat M_s - \beta(1-\rho_s)(1+r(1-\tau^k))e^{-\sigma g_y}\hat M_{s+1} = 0.$$

(b) 1 terminal-savings residual at $s = S$:
$$\hat b_{S+1}(\{\hat c_s\}) = 0.$$

This is a system of $S - E$ equations in $S - E$ unknowns. For each candidate consumption profile, the residuals are evaluated by:

1. Computing $\widehat{\Delta c}_s$ from the definition (Section 5.1).
2. Computing $\hat M_s$ from the definition (Section 5.2).
3. Computing $n_s$ from the closed-form labor FOC (Section 5.5).
4. Computing $\hat b_s$ via the budget constraint recursion (Section 6).
5. Evaluating the residual vector.

The system is solved by `scipy.optimize.root` with the Powell hybrid method, using the previous outer-loop iteration's converged consumption profile as the initial guess for warm-starting. At the first outer-loop iteration, the no-habit forward iteration provides an adequate starting point.

### 9.3 Outer Loop Unchanged in Structure

The outer SS loop in `Steady_state_equilibrium/SS_Solver.py` — iterating on the prices $r$, $\widehat{BQ}$, and the active tax rate $\tau$ — is unaffected by the habit modification. The household's `solve_steady_state` method is now an internal root-finder rather than a forward shooter, but it returns the same triple $(\hat c_s, n_s, \hat b_s)$ to the outer loop, which uses these to update prices and check convergence.

---

## 10. Boundary and Numerical Sanity Considerations

### 10.1 Positivity of $\widehat{\Delta c}_s$

The utility function $(c - hc_{\text{prev}})^{1-\sigma}/(1-\sigma)$ is well-defined only for $c - hc_{\text{prev}} > 0$. In stationary terms, this requires $\widehat{\Delta c}_s = \hat c_s - he^{-g_y}\hat c_{s-1} > 0$ for all $s \in [E+1, S]$.

This is a constraint on the *parameter region*, not a model assumption. For $h \in [0.4, 0.7]$ (the standard lifecycle-calibration range), the constraint is comfortably satisfied in any reasonable calibration because the lifecycle consumption profile changes smoothly with age. For $h$ close to 1, the constraint can bind: the agent's consumption must grow at approximately $g_y$ per period, otherwise the habit-adjusted consumption goes negative.

In numerical practice, the root-finder should be configured to handle violations gracefully by returning a large residual (e.g., $10^6$) rather than throwing an exception, allowing the optimizer to explore away from the infeasible region.

### 10.2 Limit $h \to 1$

As $h \to 1$, the habit-adjusted consumption approaches $\hat c_s - e^{-g_y}\hat c_{s-1}$. For this to remain positive, the lifecycle consumption profile must satisfy $\hat c_s > e^{-g_y}\hat c_{s-1}$ at every age, equivalently $\hat c_s/\hat c_{s-1} > e^{-g_y} \approx 1 - g_y$. This is exactly the condition that detrended consumption fall by no more than the productivity-trend rate. In the limit, the agent is forced to smooth consumption nearly perfectly along the productivity trend.

### 10.3 Interpretation of the Effective Marginal Utility $\hat M_s$

The variable $\hat M_s$ captures the *net* benefit of an additional unit of consumption at age $s$:

$$\hat M_s \;=\; \underbrace{\widehat{\Delta c}_s^{-\sigma}}_{\text{direct marginal utility}} \;-\; \underbrace{h\beta(1-\rho_s)e^{-\sigma g_y}\widehat{\Delta c}_{s+1}^{-\sigma}}_{\text{discounted future habit cost}}.$$

The first term is the marginal utility of habit-adjusted consumption today. The second term is the cost: today's consumption raises tomorrow's habit, which raises tomorrow's reference point and lowers tomorrow's marginal utility of habit-adjusted consumption. The agent internalizes this trade-off, so the *effective* marginal utility (the relevant object for optimization) is the net of the two.

A consequence is that the labor FOC uses $\hat M_s$ rather than the direct marginal utility, because the labor supply decision compares the value of an additional consumption unit (net of all future habit costs) against the disutility of working.

---

## 11. Summary of Equations That Change

For ease of cross-referencing in the thesis chapter, here is the complete list of equations that are modified by introducing internal habit formation. Equation numbers refer to the original paper (Section 2).

| Original eq. | New form |
|---|---|
| 2.3 (lifetime utility) | $U = \sum_s \beta^{s-1}\Pi_s\, u(c_s, c_{s-1}, n_s)$, includes habit argument |
| 2.4 (period utility) | $u = \frac{(c - hc_{\text{prev}})^{1-\sigma}}{1-\sigma} + e^{tg_y(1-\sigma)}\chi_s b[\cdots]^{1/\upsilon}$ |
| 2.26 (Euler) | $\hat M_s = \beta(1-\rho_s)(1+r(1-\tau^k))e^{-\sigma g_y}\hat M_{s+1}$, with $\hat M_s$ as defined in §5.2 |
| 2.27 (labor FOC, intermediate) | $\chi_s (b/\tilde l)\eta^{\upsilon-1}(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} = (1-\tau^l)\hat w \hat M_s$ |
| 2.28 (closed-form $n_s$) | $n_s = \tilde l[1+(\chi_s b/(\tilde l \hat w (1-\tau^l)\hat M_s))^{\upsilon/(\upsilon-1)}]^{-1/\upsilon}$ |

Equations 2.1, 2.2 (demography), 2.5 (budget), 2.6–2.10 (production, prices), 2.11–2.15 (market clearing, bequests, pensions), 2.16 (tax rates), and 2.17–2.25 (stationarization of all aggregates) are unchanged.

Two new definitions enter the model:

$$\widehat{\Delta c}_s = \hat c_s - h\, e^{-g_y}\,\hat c_{s-1}, \qquad \hat M_s = \widehat{\Delta c}_s^{-\sigma} - h\,\beta(1-\rho_s)\, e^{-\sigma g_y}\,\widehat{\Delta c}_{s+1}^{-\sigma},$$

with boundary conventions $\hat c_E = 0$ and $\hat M_S = \widehat{\Delta c}_S^{-\sigma}$.

The $\chi_s$ calibration formula updates by the substitution $\hat c_s^{-\sigma} \to \hat M_s$.

The equilibrium-solving algorithm changes from forward shooting (eq. 2.26's two-period structure permitted) to global root-finding (eq. 2.26's modified form has hidden three-period coupling via the definition of $\hat M_s$).

One new parameter is introduced: the habit intensity $h \in [0, 1)$. Standard literature values (Constantinides 1990, Carroll-Overland-Weil 2000) place $h$ in $[0.4, 0.7]$ for lifecycle consumption smoothing.

---

## 12. Verification Summary

The derivations above were checked at the following points:

| Check | Result |
|---|---|
| FOC w.r.t. $c_s$ reduces to standard MU at $h=0$ | ✓ ($\lambda_s = \beta^{s-1}\Pi_s c_s^{-\sigma}$) |
| Boundary FOC at $s = S$ has no future habit term | ✓ (by construction, no period $S+1$) |
| FOC w.r.t. $b_{s+1}$ is structurally unchanged | ✓ (no utility term involves $b$) |
| Labor FOC reduces to paper's eq. 2.27 at $h=0$ | ✓ ($\hat M_s = \hat c_s^{-\sigma}$) |
| Euler equation reduces to paper's eq. 2.26 at $h=0$ | ✓ ($\hat M_s = \hat c_s^{-\sigma}$) |
| Closed-form $n_s$ reduces to paper's eq. 2.28 at $h=0$ | ✓ (by substitution) |
| $\chi_s$ calibration reduces to existing formula at $h=0$ | ✓ (by substitution) |
| Algebraic identity $\eta^{\upsilon-1}(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} = (\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}$ verified | ✓ (Section 5.5) |
| $\widehat{\Delta c}_s$ stationarization yields balanced-growth-consistent form | ✓ ($e^{-g_y}$ factor captures one-period productivity trend) |
| Comparative static: $\partial n_s/\partial \chi_s < 0$ | ✓ ($\chi_s$ increase makes bracket inside parenthesis larger; outer power $-1/\upsilon < 0$ then makes $n_s$ smaller) |
| Limit $\chi_s \to \infty$: $n_s \to 0$ | ✓ |
| Limit $h \to 1$ requires $\widehat{\Delta c}_s > 0$ everywhere | ✓ (highlighted as parameter-region constraint) |

The derivations are internally consistent and the no-habit limit recovers the existing model exactly. This is the prerequisite for the implementation to satisfy in a regression test: with $h = 0$, the new code must reproduce the existing model's equilibrium quantities to numerical precision.

---

## Appendix: Algebraic Identity Used in Section 5.5

For $\eta \in (0, 1)$ and $\upsilon > 1$:

$$\eta^{\upsilon-1}\,(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} \;=\; (\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}.$$

**Proof.** Write $1 - \eta^\upsilon = \eta^\upsilon(\eta^{-\upsilon} - 1)$, valid because $\eta > 0$ so $\eta^\upsilon > 0$. Therefore:

$$(1-\eta^\upsilon)^{(1-\upsilon)/\upsilon} \;=\; \bigl[\eta^\upsilon\bigr]^{(1-\upsilon)/\upsilon} \cdot (\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} \;=\; \eta^{1-\upsilon}\,(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}.$$

Multiplying by $\eta^{\upsilon-1}$:

$$\eta^{\upsilon-1} \cdot \eta^{1-\upsilon}\,(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} \;=\; \eta^{(\upsilon-1)+(1-\upsilon)}\,(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} \;=\; \eta^0\,(\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon} \;=\; (\eta^{-\upsilon}-1)^{(1-\upsilon)/\upsilon}. \;\blacksquare$$

This identity is the algebraic engine that converts the implicit labor FOC into an explicit closed-form expression for $n_s$. It is independent of the habit modification — the same identity applies in the paper's original derivation of eq. 2.28.