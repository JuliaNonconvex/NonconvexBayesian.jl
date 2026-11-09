# First-order (gradient-enhanced) surrogates

[`FirstOrderGPSurrogate`](@ref) (constructed with
[`first_order_surrogate`](@ref) or `BayesOptOptions(first_order = true)`)
extends the zero-order surrogate with **gradient observations**: one GP per
input dimension per output is fitted on each partial derivative. The
surrogate's ChainRules rule then supplies the *predicted* gradient to
first-order sub-optimizers instead of differentiating the zero-order value
GP — the standard gradient-enhanced (co-kriging style) approach.

```julia
s = first_order_surrogate(f, x0)                    # ForwardDiff gradients
s = first_order_surrogate(f, x0; grad_source = ∇f)  # user gradient oracle
```

Gradient observations are collected at every exact-mode evaluation using
`grad_source`:

- `:forwarddiff` (default): `ForwardDiff.jacobian` of `f`,
- `:finite`: central finite differences (costs `2n` extra evaluations per
  observation),
- `:none`: no gradient observations; the predicted gradient falls back to
  the AD gradient of the value GP mean,
- a function `∇f(x)`: a user-supplied oracle returning a vector (or a matrix
  with one row per output).

At a training point the gradient GPs interpolate the observed gradient.

!!! note "Inexact-gradient scheme"
    The supplied gradient is the gradient-GP *prediction* and need not equal
    the AD gradient of the surrogate value. The sub-optimizer therefore
    receives an inexact gradient; this is the standard trade-off of
    gradient-enhanced surrogates and converges as the gradient data
    accumulates.

The gradient GPs' hyperparameters are refitted together with the value GPs
when `fit_grads = true`.
