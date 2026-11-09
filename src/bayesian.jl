using AbstractGPs, ForwardDiff, IntervalArithmetic, LinearAlgebra, Statistics
using SpecialFunctions: erf

# ================================================================
# Acquisitions
# ================================================================

"""
    AbstractAcquisition

Supertype of the acquisition functions used to form the surrogate model that
the sub-optimizer solves. The surrogate objective is minimized, so acquisition
values are negated where necessary (EI/PI).
"""
abstract type AbstractAcquisition end

"""
    LCB(η = 3.0)

Lower confidence bound acquisition `μ(x) - η σ(x)`. Larger `η` means more
exploration, smaller `η` more exploitation.
"""
struct LCB{T} <: AbstractAcquisition
    η::T
end
LCB() = LCB(3.0)

"""
    GPUCB(β = 3.0; schedule = nothing)

GP-UCB acquisition `μ(x) - β(t) σ(x)` where `t` is the number of observations
collected so far and `schedule(t)` optionally grows `β` with `t` (e.g.
`t -> sqrt(2log(t))`) to guarantee no-regret behaviour. Without a schedule,
`β` is constant.
"""
struct GPUCB{B,F} <: AbstractAcquisition
    β::B
    schedule::F
end
GPUCB(β = 3.0; schedule = nothing) = GPUCB(β, schedule)
_β(acq::GPUCB, t) = acq.schedule === nothing ? acq.β : acq.schedule(t)

"""
    EI(ξ = 0.01)

Negative expected improvement. `ξ` biases the acquisition towards more
exploration.
"""
struct EI{T} <: AbstractAcquisition
    ξ::T
end
EI() = EI(0.01)

"""
    PI(ξ = 0.01)

Negative probability of improvement.
"""
struct PI{T} <: AbstractAcquisition
    ξ::T
end
PI() = PI(0.01)

Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
φ(z) = exp(-z^2 / 2) / sqrt(2π)

# Best incumbent seen so far, based on the posterior mean at the observed
# points (robust to observation noise).
_fstar(pg, X) = minimum(mean(pg, X))

# Scalar acquisition value (smaller is better for the minimizing sub-optimizer).
function _acq_value(acq::EI, pg, X, x, t)
    m, v = mean_and_var(pg, [x])
    m, v = m[1], max(v[1], 0.0)
    σ = sqrt(v)
    fstar = _fstar(pg, X)
    if σ < eps(Float64) * max(abs(m), 1.0)
        return -max(fstar - m - acq.ξ, 0.0)
    end
    z = (fstar - m - acq.ξ) / σ
    return -(σ * (z * Φ(z) + φ(z)))
end
function _acq_value(acq::PI, pg, X, x, t)
    m, v = mean_and_var(pg, [x])
    m, v = m[1], max(v[1], 0.0)
    σ = sqrt(v)
    fstar = _fstar(pg, X)
    if σ < eps(Float64) * max(abs(m), 1.0)
        return -Float64(fstar - m - acq.ξ > 0)
    end
    return -Φ((fstar - m - acq.ξ) / σ)
end

# The `η` (or `β(t)`) multiplier of an LCB-style acquisition.
_acq_factor(acq::LCB, t) = acq.η
_acq_factor(acq::GPUCB, t) = _β(acq, t)

# The interval returned by the surrogate for the given acquisition. LCB-style
# acquisitions produce the [μ - ησ, μ + ησ] interval ("benefit of the doubt");
# point-valued acquisitions (EI, PI) produce a degenerate interval so that
# user code can still propagate them with IntervalArithmetic.
function _acq_interval(acq::Union{LCB,GPUCB}, pg, X, x, t)
    m, v = mean_and_var(pg, [x])
    m, v = m[1], max(v[1], 0.0)
    r = _acq_factor(acq, t) * sqrt(v)
    return interval(m - r, m + r)
end
function _acq_interval(acq::AbstractAcquisition, pg, X, x, t)
    return interval(_acq_value(acq, pg, X, x, t))
end

# ================================================================
# Surrogates
# ================================================================

# The posterior cache: `posterior` holds the per-output PosteriorGPs (or
# `nothing` before the first interval-mode evaluation) and `dirty` records
# whether the data or the GP changed since the cache was built. The field type
# is names-only so concrete per-output posterior types are accepted, and
# name-based getproperty dispatch stays type-stable.
const PosteriorCache = NamedTuple{(:posterior, :dirty)}

"""
    AbstractGPSurrogate

Supertype of the zero-order ([`ZeroOrderGPSurrogate`](@ref)) and first-order
([`FirstOrderGPSurrogate`](@ref)) Gaussian-process surrogates.
"""
abstract type AbstractGPSurrogate <: Function end

"""
    ZeroOrderGPSurrogate(f, x; kwargs...)

A Gaussian process surrogate for the (expensive) function `f`, initialized at
`x`. In `:exact` mode it evaluates `f` and records the observation; in
`:interval` mode it returns an interval built from the acquisition function
(by default `LCB`: `μ(x) ± η σ(x)`), whose gradient is supplied through a
ChainRules rule.

Keyword arguments:

- `kernel = SqExponentialKernel()`: the GP kernel,
- `X = [x]`, `y = [f(x)]`: initial observations,
- `noise = 1e-8`: observation noise,
- `std_multiple = 3.0`: `η` of the default `LCB` acquisition,
- `acquisition = nothing`: acquisition function; defaults to
  `LCB(std_multiple)`,
- `mode = :interval`: `:interval` or `:exact`,
- `every = 5`, `skip = 10`: refit the GP hyperparameters every `every`
  observations after the first `skip * every` observations,
- `last = :all`: only use the last `last` observations when refitting,
- `fit_prior = true`: whether to refit the hyperparameters by maximum
  likelihood,
- `fit_noise = false`: whether to include the noise in the MLE,
- `fit_max_iter = 20`: maximum iterations of the hyperparameter optimizer,
- `fit_bounds = nothing`: optional `(lb = ..., ub = ...)` bounds (scalars or
  vectors) on the hyperparameters,
- `fit_optimizer = nothing`: optional `(alg, options)` tuple used for the
  hyperparameter optimization; defaults to Ipopt via the
  `NonconvexBayesianIpoptExt` extension.
"""
mutable struct ZeroOrderGPSurrogate <: AbstractGPSurrogate
    f::Function
    gps::Vector{<:GP}
    X::Vector{Vector{Float64}}
    y::Union{Vector{Vector{Float64}},Vector{Float64}}
    noise::Float64
    acquisition::AbstractAcquisition
    mode::Symbol
    N::Int
    every::Int
    skip::Int
    last::Union{Symbol,Int}
    fit_prior::Bool
    fit_noise::Bool
    fit_max_iter::Int
    fit_bounds::Union{Nothing,NamedTuple}
    fit_optimizer::Any
    counter::Int
    cache::PosteriorCache
end
function ZeroOrderGPSurrogate(
    f,
    x;
    kernel = SqExponentialKernel(),
    X = [x],
    y = [f(x)],
    noise = 1e-8,
    std_multiple = 3.0,
    acquisition = nothing,
    mode = :interval,
    every = 5,
    skip = 10,
    last = :all,
    fit_prior = true,
    fit_noise = false,
    fit_max_iter = 20,
    fit_bounds = nothing,
    fit_optimizer = nothing,
)
    @assert noise > 0
    gps = [GP(AbstractGPs.ConstMean(0.0), kernel) for _ = 1:length(y[1])]
    N = length(y[1])
    return ZeroOrderGPSurrogate(
        f,
        gps,
        X,
        y,
        noise,
        acquisition === nothing ? LCB(std_multiple) : acquisition,
        mode,
        N,
        every,
        skip,
        last,
        fit_prior,
        fit_noise,
        fit_max_iter,
        fit_bounds,
        fit_optimizer,
        0,
        (posterior = nothing, dirty = true),
    )
end

"""
    FirstOrderGPSurrogate(f, x; kwargs...)

A gradient-enhanced (first-order) Gaussian process surrogate for the expensive
function `f`. In addition to the zero-order value GPs it fits one GP per input
dimension (per output) on gradient observations, so that first-order
sub-optimizers receive the *predicted* gradient via the ChainRules rule instead
of differentiating the value GP — the standard gradient-enhanced (co-kriging
style) approach when gradient observations are available. Note that this is an
inexact-gradient scheme: the supplied gradient is the gradient-GP prediction
and need not equal the AD gradient of the surrogate value.

Gradient observations are collected at every exact-mode evaluation using
`grad_source`:

- `:forwarddiff` (default): `ForwardDiff.jacobian` of `f`,
- `:finite`: central finite differences (costs `2n` extra evaluations per
  observation),
- `:none`: no gradient observations; the predicted gradient falls back to the
  AD gradient of the value GP mean,
- a function `∇f(x)`: a user-supplied gradient oracle returning a vector (or a
  matrix with one row per output).

All keyword arguments of [`ZeroOrderGPSurrogate`](@ref) are supported, plus:

- `grad_source = :forwarddiff`,
- `fit_grads = true`: whether to refit the gradient GPs' hyperparameters.
"""
mutable struct FirstOrderGPSurrogate <: AbstractGPSurrogate
    f::Function
    gps::Vector{<:GP}
    ∇gps::Vector{Vector{<:GP}} # per output, per input dimension
    X::Vector{Vector{Float64}}
    y::Union{Vector{Vector{Float64}},Vector{Float64}}
    ∇y::Vector{Matrix{Float64}} # per output: n_observations × n_inputs
    noise::Float64
    acquisition::AbstractAcquisition
    mode::Symbol
    N::Int
    every::Int
    skip::Int
    last::Union{Symbol,Int}
    fit_prior::Bool
    fit_noise::Bool
    fit_max_iter::Int
    fit_bounds::Union{Nothing,NamedTuple}
    fit_optimizer::Any
    grad_source::Union{Symbol,Function}
    fit_grads::Bool
    counter::Int
    cache::PosteriorCache
    ∇cache::PosteriorCache
end
function FirstOrderGPSurrogate(
    f,
    x;
    kernel = SqExponentialKernel(),
    X = [x],
    y = [f(x)],
    noise = 1e-8,
    std_multiple = 3.0,
    acquisition = nothing,
    mode = :interval,
    every = 5,
    skip = 10,
    last = :all,
    fit_prior = true,
    fit_noise = false,
    fit_max_iter = 20,
    fit_bounds = nothing,
    fit_optimizer = nothing,
    grad_source = :forwarddiff,
    fit_grads = true,
)
    @assert noise > 0
    @assert grad_source isa Symbol || grad_source isa Function
    n_in = length(x)
    gps = [GP(AbstractGPs.ConstMean(0.0), kernel) for _ = 1:length(y[1])]
    N = length(y[1])
    ∇gps = [[GP(AbstractGPs.ConstMean(0.0), kernel) for _ = 1:n_in] for _ = 1:N]
    ∇y = [zeros(0, n_in) for _ = 1:N]
    s = FirstOrderGPSurrogate(
        f,
        gps,
        ∇gps,
        X,
        y,
        ∇y,
        noise,
        acquisition === nothing ? LCB(std_multiple) : acquisition,
        mode,
        N,
        every,
        skip,
        last,
        fit_prior,
        fit_noise,
        fit_max_iter,
        fit_bounds,
        fit_optimizer,
        grad_source,
        fit_grads,
        0,
        (posterior = nothing, dirty = true),
        (posterior = nothing, dirty = true),
    )
    if grad_source !== :none
        ∇ = _grad_observation(s, x)
        for i = 1:N
            s.∇y[i] = vcat(s.∇y[i], ∇[i:i, :])
        end
    end
    return s
end

_records_gradients(::FirstOrderGPSurrogate) = true

function get_x_lb_ub(k::Kernel)
    x, un = flatten(k)
    lb = fill(-Inf, length(x))
    ub = fill(Inf, length(x))
    return x, lb, ub, un
end

# The gradient observation at `x` as an `n_outputs × n_inputs` matrix.
# Only called for surrogates that record gradients (`_records_gradients`).
function _grad_observation(s, x)
    src = s.grad_source
    if src === :forwarddiff
        return _forwarddiff_jacobian(s.f, x)
    elseif src === :finite
        return _finite_difference_jacobian(s.f, x)
    else
        g = src(x)
        return g isa AbstractVector ? reshape(collect(float.(g)), 1, :) : Matrix(g)
    end
end

function _forwarddiff_jacobian(f, x)
    if f(x) isa Real
        return reshape(ForwardDiff.gradient(f, x), 1, :)
    else
        return ForwardDiff.jacobian(f, x)
    end
end

function _finite_difference_jacobian(f, x)
    n_in = length(x)
    y = f(x)
    n_out = y isa Real ? 1 : length(y)
    J = zeros(Float64, n_out, n_in)
    for j = 1:n_in
        h = cbrt(eps(Float64)) * (1 + abs(x[j]))
        xp = copy(x); xp[j] += h
        xm = copy(x); xm[j] -= h
        yp = f(xp); ym = f(xm)
        col = (yp .- ym) ./ (2h)
        J[:, j] = col isa Real ? [col] : collect(col)
    end
    return J
end

# --- Hyperparameter fitting (maximum likelihood) ---------------------------

# Fit a group of independent GPs (one data vector per GP) by maximizing the
# log marginal likelihood. `ys` must be a vector with one data vector per GP.
# Returns the refitted GPs and noise.
function _fit_gp_group!(
    gps::Vector{<:GP},
    X,
    ys::Vector{<:Vector{<:Real}},
    noise,
    last,
    fit_noise;
    fit_max_iter = 20,
    fit_bounds = nothing,
    fit_optimizer = nothing,
)
    length(gps) == length(ys) ||
        error("Expected one data vector per GP, got $(length(ys)) for $(length(gps)) GPs.")
    xs_uns_1 = flatten.(getproperty.(gps, :mean))
    xs_lbs_ubs_uns_2 = get_x_lb_ub.(getproperty.(gps, :kernel))
    x1 = mapreduce(vcat, xs_uns_1) do x_un
        x_un[1]
    end
    x2 = mapreduce(vcat, xs_lbs_ubs_uns_2) do x_lb_ub_un
        x_lb_ub_un[1]
    end
    lb2 = mapreduce(vcat, xs_lbs_ubs_uns_2) do x_lb_ub_un
        x_lb_ub_un[2]
    end
    ub2 = mapreduce(vcat, xs_lbs_ubs_uns_2) do x_lb_ub_un
        x_lb_ub_un[3]
    end
    if fit_noise
        x = [x1; x2; noise]
        # 1e-6: the floor must sit above Ipopt's default bound_relax_factor
        # (1e-8), which would otherwise relax it to zero and let the MLE
        # drive the noise to a degenerate value.
        lb = [fill(-Inf, length(x1)); lb2; 1e-6]
        ub = [fill(Inf, length(x1)); ub2; Inf]
    else
        x = [x1; x2]
        lb = [fill(-Inf, length(x1)); lb2]
        ub = [fill(Inf, length(x1)); ub2]
    end
    if fit_bounds !== nothing
        lb .= max.(lb, _expand_to(fit_bounds.lb, length(lb)))
        ub .= min.(ub, _expand_to(fit_bounds.ub, length(ub)))
    end
    if last == :all
        _X, _ys = X, ys
    else
        _X = X[max(end - last + 1, 1):end]
        _ys = map(v -> v[max(end - last + 1, 1):end], ys)
    end
    obj(θ) = begin
        offset = 0
        n = fit_noise ? θ[end] : noise
        val = -sum(1:length(gps)) do i
            l1 = length(xs_uns_1[i][1])
            un1 = xs_uns_1[i][2]
            l2 = length(xs_lbs_ubs_uns_2[i][1])
            un2 = xs_lbs_ubs_uns_2[i][4]
            _gp = GP(un1(θ[offset+1:offset+l1]), un2(θ[offset+l1+1:offset+l1+l2]))
            offset += l1 + l2
            # Relative jitter keeps the covariance positive definite even when
            # the optimizer explores extreme kernel amplitudes, where the
            # finite-precision Cholesky would otherwise fail.
            dmax = maximum(xk -> _gp.kernel(xk, xk), _X)
            _n = n + sqrt(eps()) * max(dmax, 1.0)
            return logpdf(_gp(_X, _n), _ys[i])
        end
        return isfinite(val) ? val : 1e10
    end
    m = Model(obj)
    addvar!(m, lb, ub)
    alg, opts = _fit_algorithm(fit_optimizer, fit_max_iter)
    res = optimize(m, alg, x, options = opts)
    θ = res.minimizer
    offset = 0
    gps = map(1:length(gps)) do i
        l1 = length(xs_uns_1[i][1])
        un1 = xs_uns_1[i][2]
        l2 = length(xs_lbs_ubs_uns_2[i][1])
        un2 = xs_lbs_ubs_uns_2[i][4]
        out = GP(un1(θ[offset+1:offset+l1]), un2(θ[offset+l1+1:offset+l1+l2]))
        offset += l1 + l2
        return out
    end
    noise = fit_noise ? θ[end] : noise
    return gps, noise
end

_expand_to(v, n) = v isa Real ? fill(Float64(v), n) :
    (length(v) == n ? collect(Float64, v) :
     throw(ArgumentError("fit_bounds entries must be scalars or vectors of length $n (got $(length(v))).")))

# Sentinel so that the NonconvexBayesianIpoptExt extension can *add* a more
# specific method for the default hyperparameter optimizer (extensions cannot
# overwrite existing methods).
struct NoFitOptimizer end

_fit_algorithm(fit_optimizer::Tuple, fit_max_iter) = fit_optimizer
_fit_algorithm(::Nothing, fit_max_iter) = _fit_algorithm(NoFitOptimizer(), fit_max_iter)

# Fallback: overridden by the more specific NonconvexBayesianIpoptExt method.
function _fit_algorithm(fit_optimizer, fit_max_iter)
    error(
        "NonconvexBayesian needs a hyperparameter-optimization solver. Load " *
        "NonconvexIpopt (`using NonconvexIpopt`) or pass `fit_optimizer = (alg, options)` " *
        "to the surrogate or to `BayesOptOptions`.",
    )
end

# Convert the surrogate's observation-major storage (`y[obs]` is the vector of
# outputs at observation `obs`) to per-GP data vectors.
function _per_gp_data(y)
    if y isa Vector{<:Real}
        return [y]
    else
        return [getindex.(y, i) for i = 1:length(y[1])]
    end
end

function fit_mle!(s::AbstractGPSurrogate, gps, X, y, noise, last, fit_noise; kwargs...)
    gps, noise = _fit_gp_group!(
        gps,
        X,
        _per_gp_data(y),
        noise,
        last,
        fit_noise;
        kwargs...,
    )
    s.gps = gps
    s.noise = noise
    set_cache!(s, (posterior = s.cache.posterior, dirty = true))
    return s
end
function ChainRulesCore.rrule(::typeof(fit_mle!), s, gp, X, y, noise, last, fit_noise)
    return fit_mle!(s, gp, X, y, noise, last, fit_noise),
    _ -> begin
        return ntuple(_ -> NoTangent(), Val(8))
    end
end

# Refit all the GPs of the surrogate: the value GPs and, for first-order
# surrogates, the gradient GPs.
function fit_mle!(s::AbstractGPSurrogate)
    fit_mle!(
        s,
        s.gps,
        s.X,
        s.y,
        s.noise,
        s.last,
        s.fit_noise;
        fit_max_iter = s.fit_max_iter,
        fit_bounds = s.fit_bounds,
        fit_optimizer = s.fit_optimizer,
    )
    if s isa FirstOrderGPSurrogate && s.fit_grads && s.grad_source !== :none
        for i = 1:s.N
            n_in = size(s.∇y[i], 2)
            ys = [s.∇y[i][:, j] for j = 1:n_in]
            s.∇gps[i], _ = _fit_gp_group!(
                s.∇gps[i],
                s.X,
                ys,
                s.noise,
                s.last,
                false;
                fit_max_iter = s.fit_max_iter,
                fit_bounds = s.fit_bounds,
                fit_optimizer = s.fit_optimizer,
            )
        end
        set_∇cache!(s, (posterior = s.∇cache.posterior, dirty = true))
    end
    return s
end

# --- Observation and prediction -------------------------------------------

function (s::AbstractGPSurrogate)(x)
    if s.mode == :exact
        return _observe!(s, x)
    else
        return _predict(s, x)
    end
end

# Whether a surrogate records gradient observations in exact mode.
_records_gradients(::AbstractGPSurrogate) = false

function _observe!(s::AbstractGPSurrogate, x)
    y = s.f(x)
    if !(y isa Real ? isfinite(y) : all(isfinite, y))
        @warn "Skipping non-finite surrogate observation." maxlog = 10
        return y isa Real ? nai() : map(_ -> nai(), y)
    end
    s.X = vcat(s.X, [x])
    s.y = vcat(s.y, [y])
    set_cache!(s, (posterior = s.cache.posterior, dirty = true))
    if _records_gradients(s) && s.grad_source !== :none
        ∇ = _grad_observation(s, x)
        if all(isfinite, ∇)
            for i = 1:s.N
                s.∇y[i] = vcat(s.∇y[i], ∇[i:i, :])
            end
            if s isa FirstOrderGPSurrogate
                set_∇cache!(s, (posterior = s.∇cache.posterior, dirty = true))
            end
        else
            @warn "Skipping non-finite gradient observation." maxlog = 10
        end
    end
    s.counter += 1
    if s.fit_prior && s.counter > s.skip * s.every && (s.counter % s.every) == 0
        fit_mle!(s)
    end
    return interval.(y, y)
end

function _predict(s::AbstractGPSurrogate, x)
    if s.cache.dirty || s.cache.posterior === nothing
        posterior_cache = map(1:s.N) do i
            ys = eltype(s.y) <: Real ? s.y : getindex.(s.y, i)
            posterior(s.gps[i](s.X, s.noise), ys)
        end
        set_cache!(s, (posterior = posterior_cache, dirty = false))
    end
    X = s.X
    if eltype(s.y) <: Real
        return _acq_interval(s.acquisition, s.cache.posterior[1], X, x, s.counter)
    else
        return map(i -> _acq_interval(s.acquisition, s.cache.posterior[i], X, x, s.counter), 1:s.N)
    end
end

# Memoization only: the caches carry no gradient information.
function set_cache!(s::AbstractGPSurrogate, cache)
    s.cache = cache
    return s
end
ChainRulesCore.@non_differentiable set_cache!(s::AbstractGPSurrogate, cache)
function set_∇cache!(s::FirstOrderGPSurrogate, cache)
    s.∇cache = cache
    return s
end
ChainRulesCore.@non_differentiable set_∇cache!(s::FirstOrderGPSurrogate, cache)

function _refresh_∇cache!(s::FirstOrderGPSurrogate)
    if s.∇cache.posterior !== nothing && !s.∇cache.dirty
        return s.∇cache.posterior
    end
    posts = map(1:s.N) do i
        map(1:size(s.∇y[i], 2)) do j
            posterior(s.∇gps[i][j](s.X, s.noise), s.∇y[i][:, j])
        end
    end
    set_∇cache!(s, (posterior = posts, dirty = false))
    return posts
end

# The gradient of the posterior mean for output `i` at `x`. First-order
# surrogates predict it from the gradient GPs; zero-order surrogates
# differentiate the value GP mean.
function _mean_gradient(s::ZeroOrderGPSurrogate, pg, i, x)
    return ForwardDiff.gradient(x -> mean(pg, [x])[1], x)
end
function _mean_gradient(s::FirstOrderGPSurrogate, pg, i, x)
    if s.grad_source === :none || all(isempty, s.∇y)
        return ForwardDiff.gradient(x -> mean(pg, [x])[1], x)
    end
    posts = _refresh_∇cache!(s)
    return [mean(posts[i][j], [x])[1] for j in eachindex(posts[i])]
end

# IntervalArithmetic 1.0's interval constructor is opaque to Zygote, so give
# the surrogate callable an explicit rule. In exact mode the surrogate returns
# the raw function value wrapped as a degenerate interval, so the gradient is
# that of the wrapped function. In interval mode it returns the acquisition
# interval; the gradient is chained through the acquisition value.
function _interval_perturbation(Δ)
    Δ = ChainRulesCore.unthunk(Δ)
    Δ isa Interval && return inf(Δ), sup(Δ)
    Δ isa Real && return Δ, zero(Δ)
    Δ isa ChainRulesCore.AbstractZero && return 0.0, 0.0
    # structural tangent over the Interval fields
    b = getproperty(Δ, :bareinterval)
    b isa ChainRulesCore.AbstractZero && return 0.0, 0.0
    lo = getproperty(b, :lo)
    hi = getproperty(b, :hi)
    lo = lo isa ChainRulesCore.AbstractZero ? 0.0 : lo
    hi = hi isa ChainRulesCore.AbstractZero ? 0.0 : hi
    return lo, hi
end

function _perturbation_sum(Δ)
    lo, hi = _interval_perturbation(Δ)
    return lo + hi
end

function _call_gp_grad(s, pg, i, x, Δ)
    Δlo, Δhi = _interval_perturbation(Δ)
    acq = s.acquisition
    if acq isa Union{LCB,GPUCB}
        gmean = _mean_gradient(s, pg, i, x)
        gvar = ForwardDiff.gradient(x -> var(pg, [x])[1], x)
        _m, _v = mean_and_var(pg, [x])
        v = max(_v[1], 0.0)
        sv = sqrt(v)
        factor = _acq_factor(acq, s.counter)
        gr = sv == 0 ? zero(gvar) : factor .* gvar ./ (2sv)
        return (Δlo + Δhi) .* gmean .+ (Δhi - Δlo) .* gr
    else
        ga = ForwardDiff.gradient(x -> _acq_value(acq, pg, s.X, x, s.counter), x)
        return (Δlo + Δhi) .* ga
    end
end

function ChainRulesCore.rrule(s::AbstractGPSurrogate, x::AbstractVector)
    y = s(x)
    function surrogate_pullback(Δ)
        if s.mode == :exact
            yv = s.f(x)
            if yv isa Real
                gy = ForwardDiff.gradient(s.f, x)
                return ChainRulesCore.NoTangent(), _perturbation_sum(Δ) .* gy
            else
                J = ForwardDiff.jacobian(s.f, x)
                Δv = _perturbation_sum.(Δ)
                return ChainRulesCore.NoTangent(), J' * Δv
            end
        else
            Δs = Δ isa AbstractVector ? Δ : [Δ]
            gx = zero(x)
            for i in eachindex(Δs)
                pg = eltype(s.y) <: Real ? s.cache.posterior[1] : s.cache.posterior[i]
                gx .+= _call_gp_grad(s, pg, i, x, Δs[i])
            end
            return ChainRulesCore.NoTangent(), gx
        end
    end
    return y, surrogate_pullback
end

"""
    surrogate(f, x; kwargs...)

Constructs a [`ZeroOrderGPSurrogate`](@ref) for `f` initialized at `x`
(evaluating `f(x)` once) in interval mode.
"""
function surrogate(f, x; kwargs...)
    return ZeroOrderGPSurrogate(f, x; mode = :interval, kwargs...)
end

"""
    first_order_surrogate(f, x; kwargs...)

Constructs a [`FirstOrderGPSurrogate`](@ref) for `f` initialized at `x`
(evaluating `f(x)` and its gradient once) in interval mode.
"""
function first_order_surrogate(f, x; kwargs...)
    return FirstOrderGPSurrogate(f, x; mode = :interval, kwargs...)
end

"""
    get_lo(x)

The lower bound of an interval, mapped elementwise over arrays and vectors;
the identity for reals. Used to extract the acquisition lower bound from a
surrogate's interval output.
"""
get_lo(x::Interval) = inf(x)

# IntervalArithmetic defines no ChainRules rules, so Zygote's fallback
# differentiates `inf` through copysign/flipsign, which throws on zero
# cotangents (e.g. under NonconvexCore's row-by-row Jacobian assembly).
# Explicit rules return structural tangents (lo seeded, hi zero) so the
# surrogate gradient path never descends into those internals.
function ChainRulesCore.rrule(::typeof(inf), x::Interval)
    function inf_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        Δ isa ChainRulesCore.AbstractZero &&
            return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent())
        Δv = Δ isa Real ? Δ : 0.0
        b = bareinterval(x)
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.Tangent{typeof(x)}(
                bareinterval = ChainRulesCore.Tangent{typeof(b)}(lo = Δv, hi = zero(Δv)),
                decoration = ChainRulesCore.NoTangent(),
                isguaranteed = ChainRulesCore.NoTangent(),
            ),
        )
    end
    return inf(x), inf_pullback
end
function ChainRulesCore.rrule(::typeof(inf), x::IntervalArithmetic.BareInterval)
    function inf_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        Δ isa ChainRulesCore.AbstractZero &&
            return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent())
        Δv = Δ isa Real ? Δ : 0.0
        return (
            ChainRulesCore.NoTangent(),
            ChainRulesCore.Tangent{typeof(x)}(lo = Δv, hi = zero(Δv)),
        )
    end
    return inf(x), inf_pullback
end
# Zygote's broadcast pullback seeds `ForwardDiff.Dual` values whose partials
# are typed by the seeded value (an `Interval`); re-type them to the output
# type so the broadcast machinery propagates scalar lower-bound perturbations.
function get_lo(x::ForwardDiff.Dual)
    v = get_lo(ForwardDiff.value(x))
    p = ForwardDiff.partials(x)
    fp = ntuple(i -> _lo_seed(p[i]), length(p))
    return ForwardDiff.Dual(v, fp)
end
get_lo(x::AbstractVector) = map(get_lo, x)
get_lo(x::Real) = x
_lo_seed(p) = p isa Interval ? Float64(inf(p)) : Float64(p)
_lower_f(s) = get_lo ∘ s
_equality_f(s) = x -> begin
    t = s(x)
    return [inf.(t); .-sup.(t)]
end

# Build a surrogate (zero-, first-order or co-kriging) for a single expensive
# function. The first-order-only kwargs (grad_source, fit_grads) are consumed
# here so they are not forwarded to ZeroOrderGPSurrogate. `co_kriging` takes
# precedence over `first_order`.
function _make_surrogate(
    f,
    x;
    first_order::Bool,
    acquisition::AbstractAcquisition,
    co_kriging::Bool = false,
    grad_source = :forwarddiff,
    fit_grads = true,
    kwargs...,
)
    if co_kriging
        return CoKrigingSurrogate(
            f,
            x;
            acquisition = acquisition,
            grad_source = grad_source,
            kwargs...,
        )
    elseif first_order
        return FirstOrderGPSurrogate(
            f,
            x;
            acquisition = acquisition,
            grad_source = grad_source,
            fit_grads = fit_grads,
            kwargs...,
        )
    else
        return ZeroOrderGPSurrogate(f, x; acquisition = acquisition, kwargs...)
    end
end

"""
    surrogate_model(vecmodel::VecModel; first_order = false, acquisition = nothing,
                    std_multiple = 3.0, kwargs...)

Replaces every expensive objective/constraint function in `vecmodel` by a GP
surrogate, returning `(surrogate_model, surrogates)`. The objective surrogate
uses `acquisition` (default `LCB(std_multiple)`); constraint surrogates always
use `LCB(std_multiple)` since feasibility calls for a conservative lower
bound. When `first_order = true`, [`FirstOrderGPSurrogate`](@ref)s are used so
first-order sub-optimizers receive predicted gradients.
"""
function surrogate_model(
    vecmodel::VecModel;
    first_order = false,
    acquisition = nothing,
    std_multiple = 3.0,
    kwargs...,
)
    surrogates = Function[]
    x0 = getinit(vecmodel)
    obj_acq = acquisition === nothing ? LCB(std_multiple) : acquisition
    if :expensive in vecmodel.objective.flags
        s = _make_surrogate(
            vecmodel.objective.f,
            copy(x0);
            first_order = first_order,
            acquisition = obj_acq,
            kwargs...,
        )
        push!(surrogates, s)
        obj = Objective(_lower_f(s), vecmodel.objective.multiple, vecmodel.objective.flags)
    else
        obj = vecmodel.objective
    end
    expensive_eq_constraints = filter(vecmodel.eq_constraints.fs) do c
        :expensive in c.flags
    end
    cheap_eq_constraints = filter(vecmodel.eq_constraints.fs) do c
        @assert c isa EqConstraint
        !(:expensive in c.flags)
    end
    eq_constraints = VectorOfFunctions(cheap_eq_constraints)
    ineq_constraints1 = Tuple(
        mapreduce(vcat, expensive_eq_constraints; init = Union{}[]) do c
            @assert c isa EqConstraint
            s = _make_surrogate(
                c,
                copy(x0);
                first_order = first_order,
                acquisition = LCB(std_multiple),
                kwargs...,
            )
            push!(surrogates, s)
            return IneqConstraint(
                _equality_f(s),
                [zero.(c.rhs); zero.(c.rhs)],
                c.dim * 2,
                c.flags,
            )
        end,
    )
    ineq_constraints2 = map(vecmodel.ineq_constraints.fs) do c
        @assert c isa IneqConstraint
        if :expensive in c.flags
            s = _make_surrogate(
                c,
                copy(x0);
                first_order = first_order,
                acquisition = LCB(std_multiple),
                kwargs...,
            )
            push!(surrogates, s)
            return IneqConstraint(_lower_f(s), zero.(c.rhs), c.dim, c.flags)
        else
            return c
        end
    end
    ineq_constraints = VectorOfFunctions((ineq_constraints1..., ineq_constraints2...),)
    return VecModel(
        obj,
        eq_constraints,
        ineq_constraints,
        vecmodel.sd_constraints,
        vecmodel.box_min,
        vecmodel.box_max,
        vecmodel.init,
        vecmodel.integer,
    ),
    Tuple(surrogates)
end

"""
    update_surrogates!(model, surrogates, x)

Evaluates `model` at `x` with the surrogates in exact mode (recording the
true observations, including gradient observations for first-order
surrogates), then switches them back to interval mode. Returns the objective
value, the inequality and the equality constraint values at `x`.
"""
function update_surrogates!(model, surrogates, x)
    foreach(surrogates) do s
        s.mode = :exact
    end
    o = model.objective(x)
    i = model.ineq_constraints(x)
    e = model.eq_constraints(x)
    foreach(surrogates) do s
        s.mode = :interval
    end
    return o, i, e
end

# ================================================================
# Options, workspace, algorithm
# ================================================================

struct BayesOptOptions{S,N<:NamedTuple}
    sub_options::S
    maxiter::Int
    initialize::Bool
    ninit::Int
    init_strategy::Symbol
    ctol::Float64
    ftol::Float64
    postoptimize::Bool
    verbose::Bool
    keep_all::Bool
    acq_multistart::Int
    deduplicate::Bool
    xtol_dup::Float64
    trust_region::Bool
    tr_length_init::Float64
    tr_min_factor::Float64
    tr_success_max::Int
    tr_fail_max::Union{Int,Nothing}
    nt::N
end

"""
    BayesOptOptions(sub_options; kwargs...)

Options of the [`BayesOptAlg`](@ref) Bayesian-optimization loop. `sub_options`
(the options of the sub-optimizer) is required. The keyword arguments are:

- `maxiter = 100`: the maximum number of BO iterations (each iteration
  evaluates the expensive functions once).
- `initialize = true`, `ninit = 10`, `init_strategy = :sobol_direct`:
  initialization control; `:sobol_direct` evaluates the true functions at
  the Sobol points, `:acquisition` solves the surrogate from each point.
- `ctol = 1e-4`: feasibility tolerance; `ftol = 1e-4`: relative objective
  tolerance for convergence.
- `postoptimize = true`: polish the incumbent on the exact model after the
  loop.
- `verbose = false`, `keep_all = false`: per-iteration logging and query
  trajectory recording.
- `acq_multistart = 1`, `deduplicate = true`, `xtol_dup = 1e-8`:
  acquisition multistart and duplicate-point filtering.
- `trust_region = false`, `tr_length_init = 0.8`, `tr_min_factor = 1/32`,
  `tr_success_max = 3`, `tr_fail_max = nothing`: TuRBO-style local
  trust-region restarts (requires finite variable bounds).
- `acquisition = nothing`: the objective acquisition (default
  `LCB(std_multiple)`); constraint surrogates always use LCB.
- `std_multiple = 3.0`: the `η` of the default LCB acquisition.
- `first_order = false`, `grad_source = :forwarddiff`, `fit_grads = true`:
  first-order (gradient-enhanced) surrogates and gradient-observation
  collection.
- `kernel = SqExponentialKernel()`, `noise = 1e-8`: the GP kernel and the
  observation noise.
- `fit_prior = true`, `fit_noise = false`, `every = 2`, `skip = 2`,
  `last = :all`: hyperparameter refit control.
- `fit_max_iter = 20`, `fit_bounds = nothing`, `fit_optimizer = nothing`:
  the hyperparameter optimizer (defaults to Ipopt via the
  `NonconvexBayesianIpoptExt` extension).
"""
function BayesOptOptions(;
    sub_options,
    initialize = true,
    ninit = 10,
    init_strategy = :sobol_direct,
    maxiter = 100,
    std_multiple = 3.0,
    kernel = SqExponentialKernel(),
    noise = 1e-8,
    acquisition = nothing,
    ctol = 1e-4,
    ftol = 1e-4,
    postoptimize = true,
    verbose = false,
    keep_all = false,
    acq_multistart = 1,
    deduplicate = true,
    xtol_dup = 1e-8,
    trust_region = false,
    tr_length_init = 0.8,
    tr_min_factor = 1 / 32,
    tr_success_max = 3,
    tr_fail_max = nothing,
    fit_prior = true,
    fit_noise = false,
    every = 2,
    skip = 2,
    last = :all,
    first_order = false,
    grad_source = :forwarddiff,
    fit_grads = true,
    co_kriging = false,
    fit_max_iter = 20,
    fit_bounds = nothing,
    fit_optimizer = nothing,
)
    init_strategy in (:sobol_direct, :acquisition) ||
        error("init_strategy must be :sobol_direct or :acquisition, got $(init_strategy).")
    acq_multistart >= 1 || error("acq_multistart must be at least 1.")
    maxiter >= 1 || error("maxiter must be at least 1.")
    0 < tr_length_init <= 1 || error("tr_length_init must be in (0, 1].")
    0 < tr_min_factor < 1 || error("tr_min_factor must be in (0, 1).")
    tr_success_max >= 1 || error("tr_success_max must be at least 1.")
    return BayesOptOptions(
        sub_options,
        maxiter,
        initialize,
        ninit,
        init_strategy,
        ctol,
        ftol,
        postoptimize,
        verbose,
        keep_all,
        acq_multistart,
        deduplicate,
        xtol_dup,
        trust_region,
        tr_length_init,
        tr_min_factor,
        tr_success_max,
        tr_fail_max,
        (
            std_multiple = std_multiple,
            kernel = kernel,
            noise = noise,
            every = every,
            skip = skip,
            last = last,
            fit_prior = fit_prior,
            fit_noise = fit_noise,
            acquisition = acquisition,
            first_order = first_order,
            grad_source = grad_source,
            fit_grads = fit_grads,
            co_kriging = co_kriging,
            fit_max_iter = fit_max_iter,
            fit_bounds = fit_bounds,
            fit_optimizer = fit_optimizer,
        ),
    )
end

"""
    BayesOptWorkspace

The workspace of the [`BayesOptAlg`](@ref) meta-algorithm: the original
model, the surrogate sub-workspace, the surrogates, the callback and the
query history. `NonconvexCore.reset!` rebuilds the surrogates from scratch so
the workspace can be re-run or nested in other meta-algorithms.
"""
mutable struct BayesOptWorkspace{
    M<:VecModel,
    S1<:Workspace,
    A1,
    A2,
    X<:AbstractVector,
    O<:BayesOptOptions,
    S2<:Union{Tuple,AbstractVector},
    C,
} <: Workspace
    model::M
    sub_workspace::S1
    sub_alg::A2
    optimizer::A1
    x0::X
    options::O
    surrogates::S2
    callback::C
    history::Vector{Any}
end

"""
    BayesOptResult

The result returned by [`BayesOptAlg`](@ref). In addition to the
`GenericResult`-style fields (`minimizer`, `minimum`, `iter`,
`maxiter_reached`, `tol`, `convstate`, `fcalls`), it carries the
`sub_result` of the last surrogate solve, the trained `surrogates`, and the
query `trajectory` when `keep_all = true`.

- `fcalls` counts the expensive-function evaluations performed during this
  `optimize` call (the initial point is not counted).
- `n_restarts` counts the TuRBO-style trust-region restarts (0 when
  `trust_region = false`).
"""
mutable struct BayesOptResult{O,I,M1,M2,T,C,S1,S2,H} <: AbstractResult
    optimizer::O
    initial_x::I
    minimizer::M1
    minimum::M2
    iter::Int
    maxiter_reached::Bool
    tol::T
    convstate::C
    fcalls::Int
    n_restarts::Int
    sub_result::S1
    surrogates::S2
    trajectory::H
end

# ScaledSobolIterator adapted from BayesianOptimization.jl
struct ScaledSobolIterator{T,D}
    lowerbounds::Vector{T}
    upperbounds::Vector{T}
    N::Int
    seq::SobolSeq{D}
end

"""
    ScaledSobolIterator(lowerbounds, upperbounds, N;
                        seq = SobolSeq(length(lowerbounds)))
Returns an iterator over `N` elements of a Sobol sequence between `lowerbounds`
and `upperbounds`. The first `N` elements of the Sobol sequence are skipped for
better uniformity (see https://github.com/stevengj/Sobol.jl)
"""
function ScaledSobolIterator(
    lowerbounds,
    upperbounds,
    N;
    seq = SobolSeq(length(lowerbounds)),
)
    N > 0 && skip(seq, N)
    ScaledSobolIterator(lowerbounds, upperbounds, N, seq)
end
Base.length(it::ScaledSobolIterator) = it.N
@inline function Base.iterate(it::ScaledSobolIterator, s = 1)
    s == it.N + 1 && return nothing
    Sobol.next!(it.seq, it.lowerbounds, it.upperbounds), s + 1
end

_fcalls(surrogates) = sum(s -> s.counter, surrogates; init = 0)

# ================================================================
# TuRBO-style local trust regions
# ================================================================

"""
    _tr_bounds(center, L, lb, ub)

The bounds of the trust-region hyperrectangle with per-dimension edge lengths
`L`, centered at `center` and clamped to the variable bounds.
"""
function _tr_bounds(center, L, lb, ub)
    return max.(center .- L ./ 2, lb), min.(center .+ L ./ 2, ub)
end

"""
    _update_tr(L, improved, success, failure, τ_success, τ_fail, L_init, L_min)

The TuRBO success/failure rule: double `L` (capped at `L_init`) after
`τ_success` consecutive improvements, halve it after `τ_fail` consecutive
non-improvements, and restart (reset `L` to `L_init`) when the region
collapses below `L_min`. Returns `(L, restart, success, failure)`.
"""
function _update_tr(L, improved, success, failure, τ_success, τ_fail, L_init, L_min)
    if improved
        success += 1
        failure = 0
        L = min.(2 .* L, L_init)
        return L, false, success, failure
    else
        failure += 1
        success = 0
        if failure >= τ_fail
            L = L ./ 2
            failure = 0
            if all(L .< L_min)
                return copy(L_init), true, success, failure
            end
        end
        return L, false, success, failure
    end
end

# Evaluate the true (expensive) functions at `x` through the surrogates and
# report the objective value and feasibility.
function _evaluate!(workspace, x; ctol)
    ob, ineq, eq = update_surrogates!(workspace.sub_workspace.model, workspace.surrogates, x)
    feasible = all(ineq .<= ctol) && all(abs.(eq) .<= ctol)
    return ob, ineq, eq, feasible
end

_surrogate_objective(model, x) = begin
    v = model.objective(x)
    return v isa Real ? v : sum(v)
end

_mindist(surrogates, x) = begin
    d = Inf
    for s in surrogates
        for xk in s.X
            d = min(d, norm(xk .- x))
        end
    end
    return d
end

# Solve the acquisition (the surrogate model) restricted to the box
# `[tr_lb, tr_ub]` (the trust region, or the full variable box), with
# `nstarts` starts: the warm start plus `nstarts - 1` Sobol starts clamped
# into the box. The best candidate by surrogate objective is returned; when
# `dedup = true` candidates within `xtol_dup` of an already-evaluated point
# are skipped in favour of the next-best candidate. The subproblem bounds are
# restored before returning.
function _solve_acquisition!(
    workspace::BayesOptWorkspace,
    warm,
    sob,
    nstarts,
    dedup,
    xtol_dup,
    tr_lb,
    tr_ub,
)
    @unpack sub_workspace, surrogates = workspace
    model = sub_workspace.model
    orig_lb, orig_ub = getmin(model), getmax(model)
    setmin!(model, tr_lb)
    setmax!(model, tr_ub)
    try
        sub_workspace.x0 .= warm
        r = optimize!(sub_workspace)
        v = _surrogate_objective(model, r.minimizer)
        candidates = [(x = collect(float.(r.minimizer)), r = r, v = v)]
        if nstarts > 1 && sob !== nothing
            for x1 in Iterators.take(sob, nstarts - 1)
                x1 = clamp.(x1, tr_lb, tr_ub)  # keep the starts inside the box
                sub_workspace.x0 .= x1
                rr = optimize!(sub_workspace)
                vv = _surrogate_objective(model, rr.minimizer)
                push!(candidates, (x = collect(float.(rr.minimizer)), r = rr, v = vv))
            end
        end
        sort!(candidates, by = c -> c.v)
        if dedup && !isempty(surrogates)
            for c in candidates
                if _mindist(surrogates, c.x) >= xtol_dup
                    sub_workspace.x0 .= c.x
                    return c.r
                end
            end
            # All candidates duplicate evaluated points; keep the best one
            # anyway rather than wasting the iteration.
            sub_workspace.x0 .= candidates[1].x
            return candidates[1].r
        else
            sub_workspace.x0 .= candidates[1].x
            return candidates[1].r
        end
    finally
        setmin!(model, orig_lb)
        setmax!(model, orig_ub)
    end
end

function optimize!(workspace::BayesOptWorkspace)
    @unpack sub_workspace, surrogates = workspace
    @unpack options, x0, callback = workspace
    @unpack maxiter, ctol, ftol, postoptimize = options
    @unpack ninit, initialize, init_strategy = options
    @unpack acq_multistart, deduplicate, xtol_dup, verbose, keep_all = options
    @unpack trust_region, tr_length_init, tr_min_factor, tr_success_max, tr_fail_max = options
    smodel = sub_workspace.model
    fcalls_start = _fcalls(surrogates)
    minval = Inf
    minimizer = copy(x0)
    prevx = copy(x0)
    lb, ub = getmin(smodel), getmax(smodel)
    finite_bounds = all(isfinite, lb) && all(isfinite, ub)
    tol = Tolerance(fabs = ftol, frel = ftol, infeas = ctol)
    trust_region && !finite_bounds && error(
        "trust_region = true requires finite variable bounds for all variables.",
    )

    if initialize
        if !finite_bounds
            @warn "Skipping initialization because the variable bounds are not all finite."
        elseif init_strategy == :sobol_direct
            # Cheap standard BO initialization: evaluate the true expensive
            # functions directly at the Sobol points.
            for x1 in ScaledSobolIterator(lb, ub, ninit)
                ob, _, _, feasible = _evaluate!(workspace, x1; ctol = ctol)
                if feasible && ob <= minval
                    minval = ob
                    minimizer = copy(x1)
                end
            end
        else # :acquisition
            # Solve the surrogate from each Sobol point instead.
            x2 = x0
            for x1 in ScaledSobolIterator(lb, ub, ninit)
                sub_workspace.x0 .= x1
                r = optimize!(sub_workspace)
                if r.minimizer != x2
                    ob, _, _, feasible = _evaluate!(workspace, r.minimizer; ctol = ctol)
                    if feasible && ob <= minval
                        minval = ob
                        minimizer = copy(r.minimizer)
                    end
                end
                x2 = r.minimizer
            end
        end
        sub_workspace.x0 .= minimizer
    end

    # Sobol iterator for the acquisition multistart (persistent across
    # iterations; start points are clamped into the current trust region).
    sob = finite_bounds ? ScaledSobolIterator(lb, ub, maxiter * acq_multistart + 1) : nothing

    # TuRBO-style trust-region state: a hyperrectangle with per-dimension edge
    # lengths `L` centered on the incumbent, grown after consecutive
    # improvements, shrunk after consecutive failures, and restarted when it
    # collapses below `L_min`.
    L_init = tr_length_init .* (ub .- lb)
    L_min = tr_min_factor .* L_init
    L = copy(L_init)
    success = 0
    failure = 0
    n_restarts = 0
    τ_fail = tr_fail_max === nothing ? max(1, length(lb)) : tr_fail_max

    iter = 0
    converged = false
    convstate = ConvergenceState{Float64}()
    while iter < maxiter
        tr_lb, tr_ub = trust_region ? _tr_bounds(minimizer, L, lb, ub) : (lb, ub)
        r = _solve_acquisition!(
            workspace,
            minimizer,
            sob,
            acq_multistart,
            deduplicate,
            xtol_dup,
            tr_lb,
            tr_ub,
        )
        x = copy(r.minimizer)
        iter += 1

        prevm = minval
        ob, ineq, eq, feasible = _evaluate!(workspace, x; ctol = ctol)
        improved = feasible && ob < minval
        if feasible && ob <= minval
            minval = ob
            minimizer = copy(x)
        end
        if trust_region
            L, restart, success, failure = _update_tr(
                L,
                improved,
                success,
                failure,
                tr_success_max,
                τ_fail,
                L_init,
                L_min,
            )
            restart && (n_restarts += 1)
        end

        convstate = ConvergenceState{Float64}()
        convstate.Δx = maximum(abs.(x[j] - prevx[j]) for j = 1:length(x))
        convstate.Δf = abs(ob - prevm)
        convstate.relΔf = abs((ob - prevm) / (prevm + ftol))
        convstate.infeas =
            max(maximum(ineq; init = 0.0), 0.0, maximum(abs.(eq); init = 0.0))
        convstate.converged = feasible && abs((ob - prevm) / (prevm + ftol)) <= ftol

        if callback !== NoCallback()
            callback(
                Solution(
                    prevx,
                    x,
                    nothing,
                    prevm,
                    ob,
                    nothing,
                    [ineq; abs.(eq)],
                    nothing,
                    convstate,
                ),
            )
        end
        keep_all && push!(workspace.history, (x = copy(x), f = ob, feasible = feasible, iter = iter))
        if verbose
            @info "BayesOpt" iter = iter objective = ob feasible = feasible best = minval convergence = convstate.converged
        end
        prevx = copy(x)
        converged = convstate.converged
        if converged || iter >= maxiter
            break
        end
    end
    maxiter_reached = iter >= maxiter && !converged

    if postoptimize
        foreach(surrogates) do s
            s.mode = :exact
        end
        sub_workspace.x0 .= minimizer
        r = optimize!(sub_workspace)
        if r.minimum <= minval
            minval = r.minimum
            minimizer = copy(r.minimizer)
        end
        foreach(surrogates) do s
            s.mode = :interval
        end
    end

    fcalls = _fcalls(surrogates) - fcalls_start
    return BayesOptResult(
        workspace.optimizer,
        copy(x0),
        minimizer,
        minval,
        iter,
        maxiter_reached,
        tol,
        convstate,
        fcalls,
        n_restarts,
        r,
        surrogates,
        keep_all ? workspace.history : nothing,
    )
end

"""
    BayesOptAlg(sub_alg)

Bayesian-optimization meta-algorithm. Expensive functions (flagged with
`:expensive`) are replaced by GP surrogates, whose acquisition model is solved
by `sub_alg` to propose the next query point. The true functions are evaluated
at the candidate to update the surrogates.
"""
struct BayesOptAlg{A} <: AbstractOptimizer
    sub_alg::A
end

function NonconvexCore.solver_capabilities(alg::BayesOptAlg)
    # Expensive equality constraints are transformed into two inequality
    # constraints, so eq/ineq are always supported; sd and integer variables
    # are forwarded to the sub-solver untouched.
    sub = solver_capabilities(alg.sub_alg)
    return merge_capabilities(sub; eq = true, ineq = true, box = true)
end

function Workspace(
    model::VecModel,
    optimizer::BayesOptAlg,
    x0::AbstractVector;
    options::BayesOptOptions,
    surrogates = nothing,
    kwargs...,
)
    callback = get(kwargs, :callback, NoCallback())
    if surrogates !== nothing
        smodel = model
    else
        smodel, surrogates = surrogate_model(model; options.nt...)
    end
    checksubproblem!(
        smodel,
        optimizer.sub_alg;
        hint = "The surrogate subproblem has features that the sub-optimizer cannot handle.",
    )
    # The meta-algorithm owns the callback: it is invoked once per BO
    # iteration, so it is not forwarded to the internal surrogate solves.
    sub_kwargs = Base.structdiff(NamedTuple(kwargs), (callback = callback,))
    sub_workspace = Workspace(
        smodel,
        optimizer.sub_alg,
        x0;
        options = options.sub_options,
        sub_kwargs...,
    )
    return BayesOptWorkspace(
        model,
        sub_workspace,
        optimizer.sub_alg,
        optimizer,
        x0,
        options,
        surrogates,
        callback,
        Any[],
    )
end

function reset!(w::BayesOptWorkspace, x0 = nothing)
    x0 === nothing || (w.x0 .= x0)
    smodel, surrogates = surrogate_model(w.model; w.options.nt...)
    checksubproblem!(
        smodel,
        w.sub_alg;
        hint = "The surrogate subproblem has features that the sub-optimizer cannot handle.",
    )
    w.sub_workspace = Workspace(smodel, w.sub_alg, w.x0; options = w.options.sub_options)
    w.surrogates = surrogates
    empty!(w.history)
    return w
end
