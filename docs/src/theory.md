# Theory

This page reviews the mathematics behind `NonconvexBayesian`: Gaussian
process surrogates, acquisition functions, gradient-enhanced (co-kriging)
models, and the trust-region restart rule. The goal is to make every design
decision in the code traceable to a formula.

## The Bayesian-optimization loop

We minimize an expensive function `f` over a box-constrained (and possibly
constrained) domain. The `:expensive` functions are evaluated rarely; every
other function is evaluated exactly. Each iteration:

1. **Fit surrogates**: Gaussian processes (GPs) condition on all the
   expensive observations collected so far, `(x₁, f(x₁)), …, (x_N, f(x_N))`
   (and gradient observations, for the first-order and co-kriging
   surrogates).
2. **Solve the acquisition**: a cheaper model built from the surrogate
   posteriors — the *acquisition* — is minimized by the sub-optimizer to
   propose the next query point.
3. **Evaluate**: the true expensive functions are evaluated at the proposed
   point (recording values, and gradients when configured), the surrogates
   are updated, and the incumbent is updated when the new point is feasible
   and better.

Exploration and exploitation are balanced by the acquisition function (§
below); the surrogate posteriors supply both a prediction (mean) and an
uncertainty (variance).

## Gaussian process regression

A GP prior on `f` with constant mean `μ₀` and kernel (covariance function)
`k` assumes

```
f ~ GP(μ₀, k),    Cov(f(x), f(x')) = k(x, x').
```

The kernel encodes smoothness assumptions; `SqExponentialKernel` (the
default) gives `k(x, x') = exp(-‖x - x'‖²/2)` (in its natural units).
Observing `y = f(X) + ε` with independent Gaussian noise `ε ~ N(0, n²)`,
the posterior at a query point `x*` is Gaussian with

```
μ(x*) = μ₀ + k(x*, X) (K + n²I)⁻¹ (y - μ₀),
σ²(x*) = k(x*, x*) - k(x*, X) (K + n²I)⁻¹ k(X, x*),
```

where `K_ij = k(xᵢ, xⱼ)`. The surrogate evaluates this posterior for every
output of the expensive function (one GP per output). The posterior is
cached and only rebuilt when the observations or the hyperparameters change.

### Hyperparameter fitting

The kernel parameters, the constant mean and (optionally) the observation
noise are fitted by maximizing the log marginal likelihood

```
log p(y | X, θ) = -½ (y - μ₀)ᵀ (K + n²I)⁻¹ (y - μ₀)
                  - ½ log det(K + n²I) - (N/2) log 2π.
```

The optimization runs with a small budget (Ipopt by default, via the
`NonconvexBayesianIpoptExt` extension, or any solver passed as
`fit_optimizer`) and is guarded against non-positive-definite trial
covariances (a relative jitter keeps the Cholesky factorization valid).
Refits are scheduled by `every`/`skip`/`last` and warm-started from the
current hyperparameters.

## Intervals and the benefit of the doubt

The surrogate *callable* returns an **interval** rather than a scalar:

```
s(x) = [μ(x) - ησ(x), μ(x) + ησ(x)]      (LCB / GP-UCB),
```

where `η` (or `β(t)`) is the exploration weight. Because these intervals
propagate through user code with IntervalArithmetic, a model like
`f(x) = sqrt(expensive(x))` automatically carries the uncertainty through
the `sqrt`: the sub-optimizer minimizes the **lower bound** of the
propagated interval (`get_lo`), giving every function the "benefit of the
doubt". The surrogate's ChainRules rule supplies the gradient of this lower
bound to first-order sub-optimizers, so the acquisition solve uses exact
derivatives of the interval bounds rather than finite differences.

## Acquisition functions

The acquisition is the objective the sub-optimizer sees; since the
sub-optimizer minimizes, improvements are negated.

- **LCB (lower confidence bound)**, `μ(x) - ησ(x)`: minimizes the
  pessimistic bound; large `η` explores, small `η` exploits. This is the
  default.
- **GP-UCB**, `μ(x) - β(t)σ(x)`: as LCB, with a *schedule* `β(t)` growing
  with the observation count `t` (e.g. `t -> sqrt(2log(t + 1))`) for
  no-regret behaviour.
- **Expected improvement (EI)**, negated:
  `EI(x) = σ(x)(zΦ(z) + φ(z))` with `z = (f* - μ(x) - ξ)/σ(x)`, where
  `f*` is the best incumbent (taken as the minimum of the posterior mean at
  the observed points, which is robust to noise), `Φ`/`φ` are the standard
  normal CDF/PDF and `ξ` biases exploration.
- **Probability of improvement (PI)**, negated: `Φ((f* - μ(x) - ξ)/σ(x))`.

`LCB`/`GP-UCB` return the full `[μ - ησ, μ + ησ]` interval (uncertainty
propagates through the user's model); `EI`/`PI` return degenerate intervals
of the acquisition value. Constraint surrogates always use LCB, since
feasibility demands a conservative lower bound.

## Gradient-enhanced surrogates

Derivatives of a GP are themselves GPs: if `f ~ GP(μ₀, k)` then
`∂f/∂xⱼ ~ GP(0, ∂²k/∂xⱼ∂xⱼ')`. Gradient observations are therefore
legitimate GP data, and conditioning on them sharpens the value model
itself — this is gradient-enhanced kriging (GEK), the standard approach
when gradients are available (adjoints, automatic differentiation).

### First-order surrogates (independent gradient GPs)

`FirstOrderGPSurrogate` fits **independent** GPs: one for the values and one
per input dimension per output for the partial derivatives. The ChainRules
rule hands first-order sub-optimizers the *predicted* gradient (the gradient
GPs' means). This is an **inexact-gradient** scheme: the supplied gradient
need not equal the AD gradient of the surrogate value, but both converge to
the true gradient as the data accumulates.

### Co-kriging surrogates (joint value+gradient GP)

`CoKrigingSurrogate` conditions **one** joint GP on values and gradients
through the kernel-derivative cross-covariances:

```
K_ff[i, j]      = k(xᵢ, xⱼ),
K_f∇[i, jd]     = ∂k(xᵢ, xⱼ)/∂xⱼ_d,
K_∇∇[ic, jd]    = ∂²k(xᵢ, xⱼ)/∂xᵢ_c ∂xⱼ_d,
```

with the joint observation vector `z = [y; vec(∇yᵀ)]` and a block-diagonal
noise (observation noise `n²` on values, a small jitter `n_g²` on the exact
gradients). Because the gradient observations constrain the shape of the
value function, the value predictions improve beyond value-only kriging, and
the surrogate becomes **self-consistent**: the predicted gradient is the
exact derivative of the predicted value, so first-order sub-optimizers
receive consistent (rather than inexact) gradients. The posterior mean and
variance at `x*` are

```
μ(x*) = μ₀ + k*(x*)ᵀ (K + Λ)⁻¹ z,
σ²(x*) = k(x*, x*) - k*(x*)ᵀ (K + Λ)⁻¹ k*(x*),
```

with `k*(x*)` the joint kernel vector `[k(x*, xⱼ); ∂k(x*, xⱼ)/∂xⱼ_d]` and
`Λ` the noise matrix. The joint matrices are inherently ill-conditioned
(gradient rows are highly correlated with value rows), so a relative jitter
is added before the Cholesky factorization. The co-kriging hyperparameters
are fitted by maximum likelihood on the **value** observations (the
well-conditioned, AD-safe part of the joint likelihood); the gradients then
condition the posterior under those hyperparameters.

## Trust regions (TuRBO)

With `trust_region = true`, each acquisition solve is restricted to a
hyperrectangle centered on the incumbent with per-dimension edge length `L`,
clamped to the variable bounds. Following the TuRBO success/failure rule:

- after `tr_success_max` consecutive improvements, `L ← min(2L, L_init)`;
- after `tr_fail_max` consecutive non-improvements, `L ← L/2`;
- when `L` collapses below `tr_min_factor · L_init`, the region **restarts**
  at full size around the incumbent.

This focuses the acquisition on promising local regions — the state of the
art for high-dimensional bound-constrained Bayesian optimization — while the
restarts recover global coverage. The number of restarts is reported in the
result's `n_restarts` field.

## Convergence and protocol

The loop stops when a feasible candidate improves the incumbent by less than
`ftol` (relative) or after `maxiter` iterations. Feasibility uses `ctol` on
the true constraint evaluations. Each iteration invokes the user callback
with a `Solution` carrying the candidate, its objective value, the
constraint violations and a convergence state; `fcalls` counts the expensive
evaluations performed during the call.
