# NonconvexBayesian

[![Build Status](https://github.com/JuliaNonconvex/NonconvexBayesian.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexBayesian.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexBayesian.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexBayesian.jl)

Gaussian-process surrogate-assisted (Bayesian) optimization for
[Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl). Expensive
objective and constraint functions (flagged with `:expensive`) are replaced by
GP surrogates whose acquisition model is solved by any Nonconvex optimizer to
propose the next query point; the true functions are then evaluated at the
candidate to update the surrogates.

## Quick start

```julia
using NonconvexBayesian, NonconvexIpopt

f(x) = sqrt(x[2])
g(x, a, b) = (a*x[1] + b)^3 - x[2]

model = Model()
set_objective!(model, f, flags = [:expensive])
addvar!(model, [1e-4, 1e-4], [10.0, 10.0])
add_ineq_constraint!(model, x -> g(x, 2, 0), flags = [:expensive])
add_ineq_constraint!(model, x -> g(x, -1, 1))

alg = BayesOptAlg(IpoptAlg())
options = BayesOptOptions(
    sub_options = IpoptOptions(),
    maxiter = 50, ftol = 1e-4, ctol = 1e-5,
)
r = optimize(model, alg, [1.234, 2.345], options = options)
```

## Options

`BayesOptOptions` (the first argument `sub_options` is required):

- `sub_options`: options for the sub-optimizer.
- `maxiter`: the maximum number of Bayesian-optimization iterations (each
  iteration evaluates the expensive functions once).
- `initialize = true`: whether to initialize the surrogates before the main
  loop.
- `ninit = 10`: number of initialization points.
- `init_strategy = :sobol_direct`: `:sobol_direct` evaluates the true
  functions at the Sobol points (cheap, recommended); `:acquisition` solves
  the surrogate from each Sobol point instead.
- `ctol`: feasibility tolerance when accepting a solution.
- `ftol`: relative tolerance on the objective for convergence.
- `postoptimize = true`: run a local optimization on the exact model from the
  incumbent after the BO loop.
- `verbose = false`: print one line per iteration.
- `keep_all = false`: store the full query trajectory in the result.
- `acq_multistart = 1`: number of starts (warm start + Sobol restarts) used to
  solve the nonconvex acquisition each iteration.
- `deduplicate = true`, `xtol_dup = 1e-8`: skip candidates that are within
  `xtol_dup` of an already-evaluated point.
- `acquisition = nothing`: the acquisition function (default `LCB(3.0)`);
  see below. Constraint surrogates always use `LCB(std_multiple)`.
- `std_multiple = 3.0`: the `η` of the default LCB acquisition.
- `first_order = false`: use [`FirstOrderGPSurrogate`](@ref)s so first-order
  sub-optimizers receive predicted gradients (gradient-enhanced surrogates).
- `grad_source = :forwarddiff`: how gradient observations are collected —
  `:forwarddiff`, `:finite`, `:none` or a user gradient function.
- `fit_grads = true`: refit the gradient GPs' hyperparameters.
- `kernel = SqExponentialKernel()`: the GP kernel.
- `noise = 1e-8`: observation noise.
- `fit_prior = true`: refit hyperparameters by maximum likelihood.
- `fit_noise = false`: include the observation noise in the MLE.
- `every = 2`, `skip = 2`, `last = :all`: refit scheduling.
- `fit_max_iter = 20`: maximum iterations of the hyperparameter optimizer.
- `fit_bounds = nothing`: `(lb = ..., ub = ...)` bounds on the
  hyperparameters.
- `fit_optimizer = nothing`: `(alg, options)` tuple for the hyperparameter
  optimization; defaults to Ipopt when NonconvexIpopt is loaded.

## Acquisitions

```julia
using NonconvexBayesian
options = BayesOptOptions(
    sub_options = ...,
    acquisition = EI(0.01),      # expected improvement
)
# also: PI(0.01), GPUCB(3.0), GPUCB(1.0; schedule = t -> sqrt(2log(t))), LCB(3.0)
```

`LCB`/`GPUCB` return the `[μ - ησ, μ + ησ]` interval (uncertainty propagates
through user code with IntervalArithmetic); `EI`/`PI` return degenerate
intervals of the acquisition value. All acquisitions are negated where
necessary since the sub-optimizer minimizes.

## First-order (gradient-enhanced) surrogates

```julia
options = BayesOptOptions(sub_options = ..., first_order = true)
```

Gradient observations are recorded at every queried point (via
`grad_source`) and one GP per input dimension per output is fitted on each
partial derivative. The surrogate's ChainRules rule then supplies the
*predicted* gradient to first-order sub-optimizers instead of differentiating
the zero-order value GP — the standard gradient-enhanced kriging approach.
This is an inexact-gradient scheme: the supplied gradient is the gradient-GP
prediction and need not equal the AD gradient of the surrogate value. When a
gradient oracle is unavailable, `grad_source = :none` falls back to the AD
gradient of the value GP mean.

## Advanced: manual surrogates

```julia
s1 = surrogate(f, x0)                     # zero-order surrogate
s2 = first_order_surrogate(g, x0)         # gradient-enhanced surrogate
model = Model()
set_objective!(model, x -> get_lo(s1(x)), flags = [:expensive])
add_ineq_constraint!(model, x -> get_lo.(s2(x)), flags = [:expensive])
r = optimize(model, BayesOptAlg(IpoptAlg()), x0,
             options = BayesOptOptions(sub_options = IpoptOptions()),
             surrogates = [s1, s2])
```

The output of `s(x)` in interval mode is an `Interval` with
`lo = μ(x) - η σ(x)` and `hi = μ(x) + η σ(x)` (LCB). Use `get_lo` to extract
the lower bound. Equality constraints should be transformed to a 2-block
inequality constraint manually as `[lo; -hi]`.

## Result

`optimize` returns a `BayesOptResult` with `minimizer`, `minimum`, `iter`,
`maxiter_reached`, `tol`, `convstate`, `fcalls` (the number of expensive
evaluations performed), the last `sub_result`, the trained `surrogates`, and
the query `trajectory` when `keep_all = true`.

## Future work

TuRBO-style local trust regions with restarts (the SOTA for high-dimensional
bound-constrained BO) are the planned next step; the acquisition
candidate-selection seam in `src/bayesian.jl` is structured to support it.
