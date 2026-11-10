# Result

`optimize` returns a [`BayesOptResult`](@ref) with the `GenericResult`-style
fields:

- `minimizer`, `minimum`: the best feasible solution found,
- `iter`: the number of BO iterations run (one expensive evaluation each),
- `maxiter_reached`: whether the loop stopped at `maxiter` without
  converging,
- `tol`: the tolerances (`ftol`, `ctol`) used,
- `convstate`: the convergence state of the last candidate (feasibility,
  `Δx`, `Δf`, `relΔf`, `converged`),
- `fcalls`: the number of expensive-function evaluations performed during
  the call (the initial point is not counted),
- `n_restarts`: the number of TuRBO-style trust-region restarts,
- `initial_x`, `optimizer`,
- `sub_result`: the result of the last surrogate (acquisition) solve,
- `surrogates`: the trained surrogates (their `X`/`y` fields hold the query
  history),
- `trajectory`: the per-iteration `(x, f, feasible, iter)` records when
  `keep_all = true`.

A `callback` passed to `optimize` is invoked once per BO iteration with a
`Solution` carrying the candidate, the objective value, the constraint
violations and the convergence state. `verbose = true` logs one line per
iteration.

## Workspace

The [`BayesOptWorkspace`](@ref) underlying an optimization keeps the
original model, the surrogate sub-workspace, the surrogates, the callback
and the query history. It is created by the standard `NonconvexCore`
`Workspace` protocol and supports `reset!`:

```julia
w = Workspace(m, BayesOptAlg(IpoptAlg()), x0, options = options)
NonconvexCore.reset!(w)   # rebuild the surrogates from scratch
```

This makes the workspace reusable and nestable in other meta-algorithms
(such as multistart drivers).
