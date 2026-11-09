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
