# NEWS

## 0.2.0

### Breaking changes

- `BayesOptOptions` now requires `sub_options` (it no longer defaults to
  Ipopt options, since NonconvexIpopt is now a weak dependency). Requires
  NonconvexCore 1.7.
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
- Capability declaration via `solver_capabilities` (eq/ineq/box supported;
  semidefinite and integer features forwarded from the sub-solver).
- Hyperparameter MLE controls: `fit_max_iter`, `fit_bounds`, `fit_optimizer`.
- Exported `surrogate`, `surrogate_model`, `first_order_surrogate`, `get_lo`,
  `update_surrogates!`, `ZeroOrderGPSurrogate`, `FirstOrderGPSurrogate` and
  the acquisition types.

### Maintenance

- NonconvexIpopt demoted to a weak dependency; the
  `NonconvexBayesianIpoptExt` extension provides the default
  hyperparameter optimizer (load NonconvexIpopt or pass `fit_optimizer`).
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
