# Quick start

```julia
using NonconvexBayesian, NonconvexIpopt

f(x) = sqrt(x[2])
g(x, a, b) = (a*x[1] + b)^3 - x[2]

model = Model()
set_objective!(model, f, flags = [:expensive])
addvar!(model, [1e-4, 1e-4], [10.0, 10.0])
add_ineq_constraint!(model, x -> g(x, 2, 0), flags = [:expensive])
add_ineq_constraint!(model, x -> g(x, -1, 1))

alg = BayesOptAlg(IpoptAlg())
options = BayesOptOptions(
    sub_options = IpoptOptions(),
    maxiter = 50, ftol = 1e-4, ctol = 1e-5,
)
r = optimize(model, alg, [1.234, 2.345], options = options)
```

The `flags = [:expensive]` hint marks the functions that should be replaced
by surrogates; the remaining functions are kept and modeled natively by the
sub-optimizer (see [Constraints](constraints.md) for how expensive equality
and inequality constraints are transformed). Each iteration of the BO loop

1. solves the surrogate model (the acquisition) with the sub-optimizer,
2. evaluates the true expensive functions at the proposed point, and
3. updates the surrogates and the incumbent.

The initial surrogates are seeded by evaluating the true functions at Sobol
points (see [`BayesOptOptions`](@ref)'s `init_strategy`).

## Manual surrogates

Surrogates can also be constructed and passed in explicitly:

```julia
s1 = surrogate(f, x0)                      # zero-order surrogate
s2 = first_order_surrogate(g, x0)          # gradient-enhanced surrogate
model = Model()
set_objective!(model, x -> get_lo(s1(x)), flags = [:expensive])
add_ineq_constraint!(model, x -> get_lo.(s2(x)), flags = [:expensive])
r = optimize(model, BayesOptAlg(IpoptAlg()), x0,
             options = BayesOptOptions(sub_options = IpoptOptions()),
             surrogates = [s1, s2])
```

In interval mode `s(x)` returns an interval with `lo = μ(x) - η σ(x)` and
`hi = μ(x) + η σ(x)` (LCB); [`get_lo`](@ref) extracts the lower bound.
Equality constraints must be transformed to a 2-block inequality constraint
manually as `[lo; -hi]`.
