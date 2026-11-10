# Zero-order surrogates

[`ZeroOrderGPSurrogate`](@ref) models an expensive function `f` with one GP
per output. In `:exact` mode it evaluates `f` and records the observation; in
`:interval` mode it returns the acquisition interval (see
[Acquisitions](../acquisitions.md)). The [`surrogate`](@ref) constructor
evaluates `f` at the initial point and switches to interval mode.

```julia
s = surrogate(f, x0)                    # interval mode, LCB by default
s = surrogate(f, x0; acquisition = EI()) # expected improvement
lo = get_lo(s(x))                       # lower bound of the interval
```

The surrogate callable carries a ChainRules rule: in exact mode its gradient
is the gradient of the wrapped function; in interval mode the gradient is the
chain through the acquisition interval (the lower bound `get_lo` path).
Gradients are therefore available to first-order sub-optimizers through
Zygote without differentiating IntervalArithmetic's internals.

## Hyperparameter fitting

The GP hyperparameters are refitted by maximum likelihood every `every`
observations after the first `skip * every` observations (see
[Options](../options.md)). Non-finite observations are skipped with a
warning instead of poisoning the GP; the posterior is cached across
interval-mode evaluations and only rebuilt when the data or the GP change.

## Surrogate models and the surrogate machinery

[`AbstractGPSurrogate`](@ref) is the supertype of the zero-order
([`ZeroOrderGPSurrogate`](@ref)) and first-order
([`FirstOrderGPSurrogate`](@ref)) Gaussian-process surrogates.

[`surrogate_model`](@ref) builds the full surrogate subproblem for a
`VecModel`: it replaces every `:expensive`-flagged objective and constraint
by a GP surrogate (LCB for the objective, conservative LCB for constraints),
leaving cheap functions, bounds and semidefinite constraints untouched, and
returns `(sm, surrogates)`:

```julia
vm, x0, _ = NonconvexCore.tovecmodel(m, [1.0, 2.0])
sm, surrogates = surrogate_model(vm)             # zero-order surrogates
sm2, surrogates2 = surrogate_model(vm; first_order = true)   # gradient-enhanced
```

The returned `surrogates` are the callables whose `X`/`y` fields record the
query history. After the sub-optimizer proposes a candidate `x` from the
surrogate model, [`update_surrogates!`](@ref) evaluates the *true* model at
`x` with the surrogates in exact mode — recording the observations (values
and gradients) — and returns `(objective, ineq_constraints, eq_constraints)`
at `x`:

```julia
o, i, e = update_surrogates!(m, surrogates, x)
```

This is exactly the loop `BayesOptAlg` runs per iteration; calling the two
functions directly is useful when building custom BO loops on top of the
package.
