# Acquisitions

The surrogate model solved by the sub-optimizer is built from the posterior
mean and variance of the GP surrogates through an acquisition function,
selectable with `BayesOptOptions(acquisition = ...)` or the surrogate
constructors.

| Acquisition | Value (minimized) | Interval semantics |
|---|---|---|
| [`LCB`](@ref)`(η = 3.0)` | `μ(x) - η σ(x)` | `[μ - ησ, μ + ησ]` |
| [`GPUCB`](@ref)`(β; schedule)` | `μ(x) - β(t) σ(x)` | `[μ - β(t)σ, μ + β(t)σ]` |
| [`EI`](@ref)`(ξ = 0.01)` | `-EI(x)` | degenerate |
| [`PI`](@ref)`(ξ = 0.01)` | `-PI(x)` | degenerate |

`LCB`/`GPUCB` return the non-degenerate `[μ - ησ, μ + ησ]` interval
("benefit of the doubt"), so the uncertainty propagates through user code
with IntervalArithmetic. A larger `η`/`β` means more exploration, smaller
more exploitation.

`GPUCB` optionally grows `β` with the observation count `t` for no-regret
behaviour:

```julia
acquisition = GPUCB(1.0; schedule = t -> sqrt(2log(t + 1)))
```

`EI` and `PI` return degenerate intervals of the (negated) acquisition value,
since the sub-optimizer minimizes. `ξ` biases them towards exploration. The
best incumbent is taken as the minimum of the posterior mean at the observed
points (robust to observation noise).

Constraint surrogates always use `LCB(std_multiple)`: feasibility calls for a
conservative lower bound regardless of the objective acquisition.

## The acquisition interface

[`AbstractAcquisition`](@ref) is the supertype of all acquisitions
([`LCB`](@ref), [`GPUCB`](@ref), [`EI`](@ref), [`PI`](@ref)). An
acquisition is a callable: `acq(s)` returns the acquisition interval
(`[μ - ησ, μ + ησ]`, or a degenerate interval for EI/PI) at the current
posterior of a surrogate `s`, so it plugs into both
`BayesOptOptions(acquisition = ...)` and the surrogate constructors
`surrogate(f, x0; acquisition = ...)`.
