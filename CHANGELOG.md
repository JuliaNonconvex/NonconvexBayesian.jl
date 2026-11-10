# Changelog

## 0.3.0 (2026-11-10)

- Support NonconvexCore 2: `NonconvexCore = "2"` is now the compat lower
  bound, with the fine-grained capability declaration (`linear_eq`,
  `nonlinear_eq`, `linear_ineq`, `nonlinear_ineq`, `linear_sd`,
  `nonlinear_sd`, `box`, `continuous`, `binary`, `integer`,
  `infinite_bounds`)
- NonconvexIpopt is a regular dependency (the default hyperparameter
  optimizer); the package extension is removed
- Acquisition framework (`LCB`, `GPUCB` with a `β(t)` schedule, `EI`, `PI`),
  selectable per solve
- Zero-order, first-order (gradient-enhanced) and co-kriging (joint
  value+gradient) GP surrogates
- Direct-Sobol or acquisition-based initialization, acquisition multistart
  with duplicate-point filtering, and TuRBO-style trust-region restarts
- `GenericResult`-style results (`fcalls`, `convstate`, `maxiter_reached`,
  `iter`, `tol`, `initial_x`), callbacks, `verbose` logging, `keep_all`
  trajectories, `reset!` support
- Hyperparameter MLE controls (`fit_max_iter`, `fit_bounds`,
  `fit_optimizer`)
- DocumenterFragments documentation with a theory section, preview
  deployment and dev/stable badges
- Test suite organized with TestItems and a separate test environment;
  Aqua and JET quality checks; JuliaFormatter checks; codecov PR comments
- 100% coverage of executable source lines

## 0.1.7

- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.6 and made the package uninstallable on
  Julia 1.10/1.11.
- Allows NonconvexIpopt 0.6. Requires Julia 1.10+.
