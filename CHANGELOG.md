# Changelog

## 0.2.3

- The TuRBO trust region actually restricts the acquisition solves: the
  subproblem is solved in a freshly boxed workspace instead of mutating the
  surrogate model's bounds (most sub-solvers snapshot the bounds at workspace
  construction, so the mutation had no effect on the solve while corrupting
  the model's — and, through the shared arrays, the caller's — bounds).
- `L` doubles only after `tr_success_max` consecutive improvements, as
  documented (it previously doubled on every improvement, making
  `tr_success_max` a no-op).
- The co-kriging posterior mean now centers the value observations by the
  fitted constant prior mean `μ0`; the uncentered form was biased for any
  nonzero `μ0`.
- An observation with a non-finite gradient is now skipped entirely (value
  and gradient), keeping the value/gradient rows aligned.
- The exact-mode pullback reuses the value/gradient recorded by the forward
  pass instead of re-evaluating the expensive function on every gradient
  call.
- Hyperparameter refits are deferred while the postoptimize sub-solve is
  active, so the refit's own Ipopt solve never nests inside the sub-solver.
- The first BO iteration no longer reports `relΔf = NaN`/`Δf = Inf`.

## 0.2.2

- Allow ForwardDiff 1 and pin Zygote below 0.7.17; codecov OIDC/v5 uploads;
  docs destination persistence; logo and README polish.

## 0.2.1

- Deterministic and reproducible test runs: the runtests prelude pins BLAS
  to a single thread and seeds the global RNG; a new Reproducibility
  testitem asserts identical results across runs.
- The Expensive block constraint testitem uses postoptimize so its
  assertions no longer depend on the nonconvex hyperparameter-MLE
  trajectory across machines.

## 0.2.0 (2026-11-10)

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

## 0.1.9

- Added real-gradient and zero-gradient rrules for the surrogate callable;
  fixed `Dual` partial types and `get_lo` handling of ForwardDiff dual
  seeds; unit tests for the surrogate gradient rules and NaN returns.

## 0.1.8

- Cache GP posteriors and skip non-finite observations, with regression
  tests; added NEWS.md.

## 0.1.7

- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.6 and made the package uninstallable on
  Julia 1.10/1.11.
- Allows NonconvexIpopt 0.6. Requires Julia 1.10+.
