# NEWS

## 0.2.1

- Deterministic and reproducible test runs: the runtests prelude pins BLAS
  to a single thread and seeds the global RNG; a new Reproducibility
  testitem asserts identical results across runs.
- The Expensive block constraint testitem uses postoptimize so its
  assertions no longer depend on the nonconvex hyperparameter-MLE
  trajectory across machines.

## 0.3.0

### Breaking changes

- `BayesOptOptions` now requires `sub_options` (it no longer defaults to
  Ipopt options). Requires NonconvexCore 2; NonconvexIpopt is a regular
  dependency (the package extension was removed).
- `BayesOptResult` now carries the `GenericResult`-style fields
  (`optimizer`, `initial_x`, `iter`, `maxiter_reached`, `tol`, `convstate`,
  `fcalls`) alongside `sub_result`, `surrogates` and `trajectory`.
- `postoptimize` now defaults to `true` (matching the documentation).

### New features

- Acquisition-function framework: `LCB` (default), `GPUCB` (with a `β(t)`
  schedule), `EI` and `PI`, selectable via `BayesOptOptions(acquisition = ...)`
  or the surrogate constructors. Constraint surrogates always use LCB.
- First-order (gradient-enhanced) surrogates: `FirstOrderGPSurrogate` and
  `first_order_surrogate` fit GPs on gradient observations (ForwardDiff,
  finite differences, or a user gradient oracle) and supply the predicted
  gradient to first-order sub-optimizers via the ChainRules rule.
- Co-kriging surrogates: `CoKrigingSurrogate` and `co_kriging_surrogate`
  condition one joint GP on values and gradients through the
  kernel-derivative cross-covariances, making the predicted gradient the
  exact derivative of the predicted value.
- `init_strategy = :sobol_direct` (default) initializes the surrogates by
  evaluating the true functions at Sobol points; `:acquisition` keeps the old
  behaviour.
- Acquisition multistart (`acq_multistart`) with duplicate-point filtering
  (`deduplicate`, `xtol_dup`).
- TuRBO-style local trust-region restarts (`trust_region`): the acquisition
  is solved inside a hyperrectangle centered on the incumbent that grows
  after consecutive improvements, shrinks after consecutive failures, and
  restarts when it collapses below `tr_min_factor`; `n_restarts` is reported
  in the result.
- `callback`, `verbose` and `keep_all` (query trajectory) support in the BO
  loop; `reset!` rebuilds the surrogates so the algorithm can be nested in
  Hyperopt/Multistart.
- Capability declaration via `solver_capabilities` (linear/nonlinear
  equality and inequality constraints, box bounds and continuous variables
  supported; semidefinite, discrete and unbounded features forwarded from
  the sub-solver), aligned with NonconvexCore 2's fine-grained capability
  system.
- Hyperparameter MLE controls: `fit_max_iter`, `fit_bounds`, `fit_optimizer`.
- Exported `surrogate`, `surrogate_model`, `first_order_surrogate`, `get_lo`,
  `update_surrogates!`, `ZeroOrderGPSurrogate`, `FirstOrderGPSurrogate` and
  the acquisition types.

### Bug fixes

- The TuRBO trust region actually restricts the acquisition solves: the
  subproblem is solved in a freshly boxed workspace instead of mutating the
  surrogate model's bounds (most sub-solvers snapshot the bounds at workspace
  construction, so the mutation had no effect on the solve while corrupting
  the model's — and, through the shared arrays, the caller's — bounds).
- `L` now doubles only after `tr_success_max` *consecutive* improvements, as
  documented (it previously doubled on every improvement, making
  `tr_success_max` a no-op).
- The co-kriging posterior mean now centers the value observations by the
  fitted constant prior mean `μ0` (`μ0 + k*ᵀ(K+Λ)⁻¹(z − μ0·1_N)`); the
  uncentered form was biased for any nonzero `μ0`.
- An observation with a non-finite gradient is now skipped *entirely* (value
  and gradient), keeping the value/gradient rows aligned; previously the
  value was recorded without its gradient row and the next prediction crashed
  with a dimension mismatch.
- The exact-mode pullback reuses the value/gradient recorded by the forward
  pass instead of re-evaluating the expensive function (plus a full
  ForwardDiff gradient) on every gradient call.
- Hyperparameter refits are deferred while the postoptimize sub-solve is
  active (they run right after it), so the refit's own Ipopt solve never
  nests inside the sub-solver.
- The first BO iteration no longer reports `relΔf = NaN`/`Δf = Inf` (there is
  no previous objective value to compare against yet).
- `get_lo(::ForwardDiff.Dual)` preserves the input dual's tag (it previously
  constructed a `Dual{Nothing}`) and no longer forces non-interval partials
  through `Float64`.
- `reset!` forwards the sub-solver kwargs captured at workspace construction.
- The surrogate constructors validate the initial gradient observation
  (skipping non-finite ones, like `_observe!` does).
- `tr_fail_max = 0` is now rejected (it made the region halve on every
  iteration).

### Maintenance

- The hyperparameter optimizer defaults to Ipopt (`fit_optimizer` overrides
  it).
- DocumenterFragments documentation (`docs/`) covering the options,
  acquisitions, zero- and first-order surrogates, trust regions, capabilities
  and the result; built standalone and as a fragment of the Nonconvex docs.
- Fixed an off-by-one error in the main loop (the loop now performs exactly
  `maxiter` iterations).
- Fixed the multi-output hyperparameter-MLE path (it indexed observations
  instead of outputs).
- Made the hyperparameter MLE robust: a relative jitter keeps the covariance
  positive definite when the optimizer explores extreme kernel amplitudes,
  and the fitted-noise floor was raised to `1e-6` (above Ipopt's default
  bound relaxation, which silently erased the previous `1e-8` floor).

## 0.1.7

- Fixed invalid stdlib compat bounds (`LinearAlgebra = "1.12.0 - 1"`) that
  accidentally shipped in 0.1.6 and made the package uninstallable on
  Julia 1.10/1.11.
- Allows NonconvexIpopt 0.6. Requires Julia 1.10+.
