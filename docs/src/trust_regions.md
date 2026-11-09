# Trust regions

With `trust_region = true`, each acquisition solve is restricted to a local
hyperrectangle centered on the incumbent, with per-dimension edge length `L`
(clamped to the variable bounds). This is the TuRBO approach — the
state of the art for high-dimensional bound-constrained Bayesian
optimization.

```julia
options = BayesOptOptions(
    sub_options = ...,
    trust_region = true,
    tr_length_init = 0.8,   # initial TR edge length (fraction of the box)
    tr_min_factor = 1/32,   # restart below this fraction of the initial TR
    tr_success_max = 3,     # grow after this many consecutive improvements
    tr_fail_max = nothing,  # shrink after this many failures (default
                            # max(1, n) where n is the number of variables)
)
```

The success/failure rule:

- `L` doubles (up to `tr_length_init`) after `tr_success_max` consecutive
  improvements,
- `L` halves after `tr_fail_max` consecutive non-improvements,
- when the region collapses below `tr_min_factor * L_init`, it **restarts**
  at full size around the incumbent.

The number of restarts is reported in the result's `n_restarts` field.
Trust regions require finite variable bounds for all variables.

The acquisition multistart points (`acq_multistart`) are clamped into the
current trust region. Batched Thompson sampling over multiple simultaneous
trust regions remains a future refinement.
