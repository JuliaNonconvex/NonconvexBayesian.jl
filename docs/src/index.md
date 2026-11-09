# NonconvexBayesian

`NonconvexBayesian` provides Gaussian-process surrogate-assisted (Bayesian)
optimization for the Nonconvex ecosystem. Expensive objective and constraint
functions (flagged with `:expensive`) are replaced by GP surrogates whose
acquisition model is solved by any Nonconvex optimizer to propose the next
query point; the true functions are then evaluated at the candidate to update
the surrogates.

This documentation is a fragment of the [Nonconvex documentation](https://julianonconvex.github.io/Nonconvex.jl/)
and is also built standalone.

## Features

- [`BayesOptAlg`](@ref) meta-algorithm: wraps any Nonconvex optimizer
  (`BayesOptAlg(IpoptAlg())`, `BayesOptAlg(NLoptAlg(...))`, ...).
- Acquisition functions: [`LCB`](@ref) (default), [`GPUCB`](@ref) (with a
  `β(t)` schedule), [`EI`](@ref) and [`PI`](@ref); constraint surrogates
  always use `LCB` for conservative feasibility.
- Zero-order ([`ZeroOrderGPSurrogate`](@ref)), first-order
  ([`FirstOrderGPSurrogate`](@ref)) and co-kriging
  ([`CoKrigingSurrogate`](@ref), joint value+gradient) surrogates.
- Direct-Sobol or acquisition-based initialization, acquisition multistart
  with duplicate-point filtering, and TuRBO-style local trust-region
  restarts for high-dimensional problems.
- Full protocol integration: `GenericResult`-style results with `fcalls`,
  convergence state, callbacks, `verbose` logging, `keep_all` query
  trajectories and `reset!` support for meta-algorithm nesting.
- Solver capability declarations forwarded to the sub-solver
  (`solver_capabilities`).

See [Quick start](quickstart.md) for a complete example.
