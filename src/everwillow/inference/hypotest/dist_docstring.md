# Asymptotic distributions for test statistics

Mathematical reference for the asymptotic formulae implemented in
`distributions.py`. All results from Cowan et al. [1].

## Background: σ from the Asimov dataset

The asymptotic formulae require σ, the standard deviation of μ̂ (assumed
Gaussian with mean μ').

Wald showed that

$$
t_{\mu} = -2\ln\lambda(\mu)
= \frac{(\mu - \hat{\mu})^{2}}{\sigma^{2}}
+ \mathcal{O}\!\left(\frac{1}{\sqrt{N}}\right),
$$

so the test statistic follows a non-central χ² distribution (Eq. 19) with
non-centrality parameter Λ = (μ − μ')²/σ².

σ can be estimated from the **Asimov dataset** — the dataset where all
estimators equal their true values (μ̂ = μ', θ̂ = θ). Evaluating the test
statistic on Asimov data gives

$$
t_{\mu,A} \approx \frac{(\mu - \mu')^{2}}{\sigma^{2}},
\qquad
\sigma_{A} = \frac{|\mu - \mu'|}{\sqrt{t_{\mu,A}}}.
$$


## 1. t_μ — two-sided (Eq. 10, 38)

$$
t_\mu = -2\ln\lambda(\mu) \approx \frac{(\mu - \hat\mu)^{2}}{\sigma^{2}}.
$$

**CDF:**

$$
F(t_\mu \mid \mu') = \Phi\!\left(\sqrt{t_\mu} + \frac{\mu - \mu'}{\sigma}\right) + \Phi\!\left(\sqrt{t_\mu} - \frac{\mu - \mu'}{\sigma}\right) - 1.
$$

**p-value** (μ' = μ): p = 2(1 − Φ(√t_μ)).


## 2. t̃_μ — two-sided with physical bound (Eq. 16, 40, 44)

Imposes μ ≥ 0. When μ̂ < 0, the denominator is evaluated at μ = 0.

**CDF** (piecewise at t̃ = μ²/σ²):

- t̃ ≤ μ²/σ²: same as t_μ CDF
- t̃ > μ²/σ²: Φ(√t̃ + δ) + Φ((t̃ + μ²/σ²)/(2μ/σ) − δ) − 1, where δ = (μ−μ')/σ

**p-value** (μ' = μ):
- t̃ ≤ μ²/σ²: p = 2(1 − Φ(√t̃))
- t̃ > μ²/σ²: p = 2 − Φ(√t̃) − Φ((t̃ + q_A)/(2√q_A))


## 3. q₀ — discovery (Eq. 12, 47, 49)

$$
q_0 = \begin{cases} -2\ln\lambda(0) & \hat\mu \ge 0, \\ 0 & \hat\mu < 0. \end{cases}
$$

**CDF:** F(q₀ | μ') = Φ(√q₀ − μ'/σ).

**p-value** (μ' = 0): p = 1 − Φ(√q₀).

**Significance:** Z₀ = √q₀.


## 4. q_μ — upper limit (Eq. 14, 57)

$$
q_\mu = \begin{cases} -2\ln\lambda(\mu) & \hat\mu \le \mu, \\ 0 & \hat\mu > \mu. \end{cases}
$$

**CDF:** F(q_μ | μ') = Φ(√q_μ − (μ−μ')/σ).

**p-value** (μ' = μ): p = 1 − Φ(√q_μ).

**Upper limit:** μ_up satisfies p_μ = α, giving μ_up = μ̂ + σΦ⁻¹(1−α)
(found numerically since σ depends on μ).


## 5. q̃_μ — upper limit with physical bound (Eq. 16+14, 64, 65)

Same as q_μ but uses the physical-bound likelihood ratio t̃.

**CDF** (piecewise at q̃ = μ²/σ²):

- 0 < q̃ ≤ μ²/σ²: F = Φ(√q̃ − (μ−μ')/σ)
- q̃ > μ²/σ²: F = Φ((q̃ − (μ²−2μμ')/σ²) / (2μ/σ))

**p-value** (μ' = μ):
- 0 < q̃ ≤ q_A: p = 1 − Φ(√q̃)
- q̃ > q_A: p = 1 − Φ((q̃ + q_A)/(2√q_A))

**Significance:**
- 0 < q̃ ≤ q_A: Z = √q̃
- q̃ > q_A: Z = (q̃ + q_A)/(2√q_A) (note: not the same formula)


## Expected sensitivity

The median significance under the alternative can be read directly from
the Asimov test statistic — no Monte Carlo needed:

- q₀: med[Z₀ | μ'] = √q_{0,A}
- q_μ: med[Z_μ | 0] = √q_{μ,A}
- q̃_μ: med[Z_μ | 0] = √q̃_{μ,A}


## References

[1] G. Cowan, K. Cranmer, E. Gross and O. Vitells,
"Asymptotic formulae for likelihood-based tests of new physics",
Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727.

[2] G. J. Feldman and R. D. Cousins,
"Unified approach to the classical statistical analysis of small signals",
Phys. Rev. D 57 (1998) 3873.
