# Co-kriging surrogates

[`CoKrigingSurrogate`](@ref) (constructed with
[`co_kriging_surrogate`](@ref) or `BayesOptOptions(co_kriging = true)`)
conditions a **single joint Gaussian process** on value and gradient
observations through the kernel-derivative cross-covariances — see
[Theory](../theory.md) for the formulas. It is the consistent upgrade of the
independent [first-order surrogates](first_order.md): the predicted gradient
is the exact derivative of the predicted value, and gradient data sharpens
the value model itself.

```julia
s = co_kriging_surrogate(f, x0)                    # ForwardDiff gradients
s = co_kriging_surrogate(f, x0; grad_source = ∇f)  # user gradient oracle
```

Gradient observations are collected at every exact-mode evaluation using
`grad_source` (`:forwarddiff` by default, `:finite`, `:none` or a user
function), exactly as for [`FirstOrderGPSurrogate`](@ref). The joint
posterior implements the same prediction interface (mean, variance,
mean-and-variance) and plugs into the same acquisition machinery; all
acquisitions (`LCB`, `GPUCB`, `EI`, `PI`) work unchanged.

At a training point the joint model interpolates both the observed values
and the observed gradients, and — unlike the independent first-order model —
its value mean's derivative reproduces the observed gradients, so the
rrule-supplied gradient matches finite differences of the surrogate value.

The hyperparameters (kernel, constant mean, optionally the observation
noise) are fitted by maximum likelihood on the value observations; the
gradient observations then condition the posterior under those
hyperparameters. The joint covariance is stabilized with a relative jitter
before factorization, since gradient rows are highly correlated with value
rows.

```julia
options = BayesOptOptions(
    sub_options = ...,
    co_kriging = true,          # joint value+gradient surrogates
    grad_source = :forwarddiff, # gradient observations via ForwardDiff
)
```
