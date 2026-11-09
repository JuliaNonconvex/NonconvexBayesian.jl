# Solver capabilities

[`BayesOptAlg`](@ref) declares its capabilities through
`solver_capabilities`:

- `eq = true`: expensive equality constraints are transformed into two
  inequality constraints,
- `ineq = true`,
- `box = true`,
- `sd` and `integer` (and `infinite_bounds`) are **forwarded** from the
  sub-solver, since the surrogate machinery passes those model features
  through untouched.

For example `BayesOptAlg(IpoptAlg())` inherits `sd = false`,
`integer = false` and `infinite_bounds = true` from Ipopt's declaration, so
models with semidefinite constraints or integer variables error with an
informative message before optimization starts. The surrogate subproblem is
validated against the sub-solver's declaration via
`checksubproblem!`.
