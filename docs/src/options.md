# Options

[`BayesOptOptions`](@ref) controls the BO loop. The first argument
`sub_options` (options for the sub-optimizer) is required.

```julia
options = BayesOptOptions(
    sub_options = IpoptOptions(),   # required: sub-optimizer options
    maxiter = 100,                  # maximum BO iterations (one expensive
                                    # evaluation per iteration)
    initialize = true,              # seed the surrogates before the loop
    ninit = 10,                     # number of initialization points
    init_strategy = :sobol_direct,  # :sobol_direct | :acquisition
    ctol = 1e-4,                    # feasibility tolerance
    ftol = 1e-4,                    # relative objective tolerance
    postoptimize = true,            # polish the incumbent on the exact model
    verbose = false,                # log one line per iteration
    keep_all = false,               # store the full query trajectory
    acq_multistart = 1,             # acquisition starts (warm + Sobol)
    deduplicate = true,             # skip candidates near evaluated points
    xtol_dup = 1e-8,                # duplicate-distance tolerance
    trust_region = false,           # TuRBO-style trust-region restarts
    tr_length_init = 0.8,           # initial TR edge length (fraction of box)
    tr_min_factor = 1/32,           # restart below this fraction of the
                                    # initial TR length
    tr_success_max = 3,             # grow after this many improvements
    tr_fail_max = nothing,          # shrink after this many failures
                                    # (defaults to max(1, n))
    acquisition = nothing,          # acquisition function (default LCB(3.0))
    std_multiple = 3.0,             # η of the default LCB
    first_order = false,            # gradient-enhanced surrogates
    grad_source = :forwarddiff,     # :forwarddiff | :finite | :none | ∇f
    fit_grads = true,               # refit the gradient GPs' hyperparameters
    co_kriging = false,             # joint value+gradient (co-kriging)
                                    # surrogates; takes precedence over
                                    # first_order
    kernel = SqExponentialKernel(), # the GP kernel
    noise = 1e-8,                   # observation noise
    fit_prior = true,               # refit hyperparameters by maximum
                                    # likelihood
    fit_noise = false,              # include the noise in the MLE
    every = 2, skip = 2, last = :all,  # refit scheduling
    fit_max_iter = 20,              # hyperparameter-optimizer iterations
    fit_bounds = nothing,           # (lb = ..., ub = ...) hyperparameter
                                    # bounds
    fit_optimizer = nothing,        # (alg, options) for the MLE; defaults
                                    # to Ipopt via the extension
)
```

## Initialization

- `:sobol_direct` (default): evaluate the true expensive functions at `ninit`
  Sobol points. Cheap and standard BO practice.
- `:acquisition`: solve the surrogate from each Sobol point instead.

Initialization is skipped (with a warning) when the variable bounds are not
all finite.

## Surrogate refitting

The GP hyperparameters are refitted by maximum likelihood every `every`
evaluations after the first `skip * every` evaluations, using the last `last`
observations when `last` is an integer. `fit_noise` adds the observation
noise to the fitted parameters (with a floor of `1e-6`). The optimizer used
for the MLE is `fit_optimizer`, or Ipopt (with `fit_max_iter` iterations)
when NonconvexIpopt is loaded; otherwise an error suggests loading
NonconvexIpopt or passing `fit_optimizer`.
