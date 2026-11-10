# Constrained optimization

`BayesOptAlg` handles constraints in three ways, depending on the `flags`
hint on each constraint:

- `:expensive` constraints are replaced by GP surrogates (like expensive
  objectives), and the sub-optimizer solves the surrogate constraints;
- cheap (non-`:expensive`) constraints are modeled **natively** — the
  sub-optimizer evaluates the exact functions;
- semidefinite constraints and integer/binary variables are **passed
  through untouched** to the sub-solver (validated against the
  sub-solver's declared capabilities).

## Expensive inequality constraints

An expensive inequality constraint `g(x) ≤ 0` is replaced by the
conservative lower confidence bound of its surrogate,

```
μ_g(x) - ησ_g(x) ≤ 0,
```

so the sub-optimizer gives feasibility the benefit of the doubt exactly like
the objective. At every iteration the *true* constraint is evaluated at the
proposed point, and the candidate is accepted as feasible only when the
true violations are within `ctol` — the surrogate bound never weakens the
final feasibility test.

## Expensive equality constraints

An expensive equality constraint `h(x) = 0` is transformed into a **2-block
inequality constraint** on the surrogate interval `[lo, hi]`:

```
lo ≤ 0 ≤ hi   ⇔   [lo; -hi] ≤ 0,
```

i.e. the sub-optimizer sees `μ_h(x) - ησ_h(x) ≤ 0` and
`μ_h(x) + ησ_h(x) ≥ 0` simultaneously. The block transformation is
implemented by `_equality_f` (values `[inf; -sup]` of the interval output,
differentiable through the package's own ChainRules rules).

## Initialization, feasibility and acceptance

- The initialization (`:sobol_direct` or `:acquisition`) evaluates the true
  expensive functions (including expensive constraints) so the constraint
  surrogates start with real feasibility data.
- A candidate is accepted as the incumbent only when it is feasible at
  `ctol` on the *true* constraints and improves the objective.
- `postoptimize = true` polishes the incumbent on the **exact** model, so
  the final result satisfies the true constraints to the sub-optimizer's
  tolerance.

## Capability validation

The surrogate subproblem (with the transformed constraints) is validated
against the sub-solver's capabilities via `checksubproblem!` before
optimization: a sub-solver that cannot handle, say, semidefinite
constraints or integer variables produces an informative error instead of
silently ignoring them. See [Solver capabilities](capabilities.md).
