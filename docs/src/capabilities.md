# Solver capabilities

[`BayesOptAlg`](@ref) declares its capabilities through
`solver_capabilities`:

- `linear_eq = true`, `nonlinear_eq = true`: expensive equality constraints
  are transformed into two inequality constraints (of any linearity),
- `linear_ineq = true`, `nonlinear_ineq = true`,
- `box = true`,
- `continuous = true`: the surrogate machinery models variables as
  continuous,
- the semidefinite constraints (`linear_sd`, `nonlinear_sd`), the discrete
  variable types (`binary`, `integer`) and `infinite_bounds` are
  **forwarded** from the sub-solver, since the surrogate machinery passes
  those model features through untouched.

For example `BayesOptAlg(IpoptAlg())` inherits `linear_sd = false`,
`nonlinear_sd = false`, `binary = false`, `integer = false` and
`infinite_bounds = true` from Ipopt's declaration, so models with
semidefinite constraints or integer variables error with an informative
message before optimization starts. The surrogate subproblem is validated
against the sub-solver's declaration via `checksubproblem!`.
