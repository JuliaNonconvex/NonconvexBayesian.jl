# ================================================================
# Co-kriging: a joint GP over values and gradients
# ================================================================

"""
    CoKrigingPosterior

The posterior of a co-kriging (gradient-enhanced) Gaussian process over one
scalar output, conditioned jointly on value observations `y` and gradient
observations `∇y` through the kernel-derivative cross-covariances:

```
K_ff[i,j] = k(xᵢ, xⱼ)
K_f∇[i,jd] = ∂k(xᵢ, xⱼ)/∂xⱼ_d
K_∇∇[ic,jd] = ∂²k(xᵢ, xⱼ)/∂xᵢ_c ∂xⱼ_d'
```

Since derivatives of a GP are GPs, the gradient predictions are the exact
derivatives of the value predictions: conditioning on gradient observations
sharpens the value model and makes the surrogate self-consistent.
`CoKrigingPosterior` implements `AbstractGPs.mean`, `AbstractGPs.var` and
`AbstractGPs.mean_and_var` so the acquisition machinery can consume it
transparently.
"""
struct CoKrigingPosterior
    k::Any
    μ0::Float64
    X::Vector{Vector{Float64}}
    n_in::Int
    zv::Vector{Float64}   # value observations
    Z∇::Matrix{Float64}   # n_obs × n_in gradient observations
    L::Cholesky{Float64,Matrix{Float64}}  # Cholesky of (K + Λ)
    α::Vector{Float64}    # (K + Λ)⁻¹ z
    nval::Float64         # observation noise on values
    ngrad::Float64        # jitter on gradient observations
end

# Covariance between a value at `a` and the d-th partial derivative (with
# respect to the second argument) at `b`. ForwardDiff-based: used in the
# prediction path, where the derivatives are taken with respect to the query
# point itself (and the kernel parameters are fixed).
_value_grad_cov_q(k, a, b) = ForwardDiff.gradient(y -> k(a, y), b)

# The n×n matrix of second cross-derivatives ∂²k(a, b)/∂a_c ∂b_d (query path).
_grad_grad_cov_q(k, a, b) = ForwardDiff.jacobian(x -> _value_grad_cov_q(k, x, b), a)

# The joint observation vector: values followed by gradients (row-major).
function _joint_observations(y, ∇y)
    return vcat(y, vec(Matrix(∇y')))
end

# The joint covariance matrix with the ordering
# [values (N); gradients (N·n, row-major)], built functionally (no mutation)
# so that AD can differentiate through it.
function _joint_covariance(k, X, n_in)
    N = length(X)
    n = n_in
    Kff = [k(X[i], X[j]) for i = 1:N, j = 1:N]
    # Kfg[i, (j-1)n + d] = ∂k(Xᵢ, Xⱼ)/∂Xⱼ_d
    Kfg = reshape(
        [_value_grad_cov_q(k, X[i], X[j])[d] for j = 1:N for d = 1:n for i = 1:N],
        N,
        N * n,
    )
    # Kgg[(i-1)n + c, (j-1)n + d] = ∂²k(Xᵢ, Xⱼ)/∂Xᵢ_c ∂Xⱼ_d
    Kgg = reshape(
        [_grad_grad_cov_q(k, X[i], X[j])[c, d] for j = 1:N for d = 1:n for i = 1:N for c = 1:n],
        N * n,
        N * n,
    )
    return vcat(hcat(Kff, Kfg), hcat(Kfg', Kgg))
end

function CoKrigingPosterior(k, μ0, X, y, ∇y, nval, ngrad)
    N = length(X)
    n = length(X[1])
    size(∇y) == (N, n) ||
        error("Expected a $N×$n gradient-observation matrix, got $(size(∇y)).")
    z = _joint_observations(y, ∇y)
    K = _joint_covariance(k, X, n)
    Λ = diagm(vcat(fill(nval^2, N), fill(ngrad^2, N * n)))
    dmax = maximum(abs, diag(K))
    # GEK covariance matrices are inherently ill-conditioned (gradient rows
    # are highly correlated with value rows); a relative jitter keeps the
    # Cholesky numerically positive definite.
    A = Symmetric(K + Λ) + sqrt(eps()) * max(dmax, 1.0) * I
    L = cholesky(A; check = false)
    α = L \ z
    return CoKrigingPosterior(k, μ0, X, n, y, ∇y, L, α, nval, ngrad)
end

# The joint kernel vector k*(x): [k(x, xⱼ); ∂k(x, xⱼ)/∂xⱼ_d]. The element
# type follows the query point `x` so AD (ForwardDiff) can differentiate the
# predictions.
function _joint_kstar(pg::CoKrigingPosterior, x)
    k = pg.k
    N = length(pg.X)
    n = pg.n_in
    kstar = similar(x, N * (1 + n))
    for j = 1:N
        kstar[j] = k(x, pg.X[j])
        g = _value_grad_cov_q(k, x, pg.X[j])
        for d = 1:n
            kstar[N + (j - 1) * n + d] = g[d]
        end
    end
    return kstar
end

function AbstractGPs.mean(pg::CoKrigingPosterior, Xq::AbstractVector{<:AbstractVector})
    return map(Xq) do x
        return pg.μ0 + dot(_joint_kstar(pg, x), pg.α)
    end
end
AbstractGPs.mean(pg::CoKrigingPosterior, x::AbstractVector{<:Real}) =
    AbstractGPs.mean(pg, [x])[1]

function AbstractGPs.var(pg::CoKrigingPosterior, Xq::AbstractVector{<:AbstractVector})
    return map(Xq) do x
        kx = _joint_kstar(pg, x)
        return pg.k(x, x) - sum(abs2, pg.L \ kx)
    end
end
AbstractGPs.var(pg::CoKrigingPosterior, x::AbstractVector{<:Real}) =
    AbstractGPs.var(pg, [x])[1]

function AbstractGPs.mean_and_var(pg::CoKrigingPosterior, Xq::AbstractVector{<:AbstractVector})
    return (AbstractGPs.mean(pg, Xq), AbstractGPs.var(pg, Xq))
end

"""
    CoKrigingSurrogate(f, x; kwargs...)

A co-kriging (gradient-enhanced) Gaussian process surrogate for the
expensive function `f`, initialized at `x`. Values and gradient observations
are conditioned **jointly** through the kernel-derivative cross-covariances
(see `CoKrigingPosterior`), so the gradient predictions are the
exact derivatives of the value predictions: the surrogate is
self-consistent, and gradient data sharpens the value model itself.

Gradient observations are collected at every exact-mode evaluation using
`grad_source` (`:forwarddiff` by default, `:finite`, `:none` or a user
gradient oracle), as for [`FirstOrderGPSurrogate`](@ref). All keyword
arguments of [`ZeroOrderGPSurrogate`](@ref) are supported, plus:

- `grad_source = :forwarddiff`,
- `grad_noise = 1e-8`: the jitter on the gradient observations (gradients
  from AD are treated as exact up to this jitter).

The hyperparameters (kernel, constant mean, optionally the observation
noise) are fitted by maximum likelihood on the value observations; the
gradient observations then condition the joint posterior under those
hyperparameters.
"""
mutable struct CoKrigingSurrogate <: AbstractGPSurrogate
    f::Function
    kernel::Any
    mean_const::Float64
    X::Vector{Vector{Float64}}
    y::Union{Vector{Vector{Float64}},Vector{Float64}}
    ∇y::Vector{Matrix{Float64}} # per output: n_obs × n_in
    noise::Float64
    grad_noise::Float64
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
    counter::Int
    cache::PosteriorCache
end
function CoKrigingSurrogate(
    f,
    x;
    kernel = SqExponentialKernel(),
    X = [x],
    y = [f(x)],
    noise = 1e-8,
    grad_noise = 1e-8,
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
)
    @assert noise > 0
    @assert grad_noise > 0
    @assert grad_source isa Symbol || grad_source isa Function
    n_in = length(x)
    N = length(y[1])
    ∇y = [zeros(0, n_in) for _ = 1:N]
    s = CoKrigingSurrogate(
        f,
        kernel,
        0.0,
        X,
        y,
        ∇y,
        noise,
        grad_noise,
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
        0,
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

_records_gradients(::CoKrigingSurrogate) = true

"""
    co_kriging_surrogate(f, x; kwargs...)

Constructs a [`CoKrigingSurrogate`](@ref) for `f` initialized at `x`
(evaluating `f(x)` and its gradient once) in interval mode.
"""
function co_kriging_surrogate(f, x; kwargs...)
    return CoKrigingSurrogate(f, x; mode = :interval, kwargs...)
end

function _predict(s::CoKrigingSurrogate, x)
    if s.cache.dirty || s.cache.posterior === nothing
        posts = map(1:s.N) do i
            ys = eltype(s.y) <: Real ? s.y : getindex.(s.y, i)
            return CoKrigingPosterior(s.kernel, s.mean_const, s.X, ys, s.∇y[i], s.noise, s.grad_noise)
        end
        set_cache!(s, (posterior = posts, dirty = false))
    end
    if eltype(s.y) <: Real
        return _acq_interval(s.acquisition, s.cache.posterior[1], s.X, x, s.counter)
    else
        return map(i -> _acq_interval(s.acquisition, s.cache.posterior[i], s.X, x, s.counter), 1:s.N)
    end
end

# The gradient of the co-kriging value mean: by construction this is the
# joint-model gradient (the exact derivative of the predicted value), so the
# surrogate is self-consistent.
function _mean_gradient(s::CoKrigingSurrogate, pg, i, x)
    return ForwardDiff.gradient(x -> AbstractGPs.mean(pg, x), x)
end

# Window the observations for the hyperparameter fit.
function _window_cokriging_data(s::CoKrigingSurrogate, last)
    ys = _per_gp_data(s.y)
    if last == :all
        return s.X, ys, s.∇y
    else
        idx = max(length(s.X) - last + 1, 1):length(s.X)
        return s.X[idx], map(v -> v[idx], ys), map(∇ -> ∇[idx, :], s.∇y)
    end
end

# The co-kriging hyperparameters are fitted by maximum likelihood on the
# value observations only (the value likelihood is the well-conditioned,
# AD-safe part of the joint likelihood); the gradient observations then
# condition the joint posterior under those hyperparameters.
function fit_mle!(s::CoKrigingSurrogate)
    gps = [GP(AbstractGPs.ConstMean(0.0), s.kernel) for _ = 1:s.N]
    gps, noise = _fit_gp_group!(
        gps,
        s.X,
        _per_gp_data(s.y),
        s.noise,
        s.last,
        s.fit_noise;
        fit_max_iter = s.fit_max_iter,
        fit_bounds = s.fit_bounds,
        fit_optimizer = s.fit_optimizer,
    )
    # One shared kernel and constant mean for all outputs (from the first
    # output's fitted GP).
    s.kernel = gps[1].kernel
    s.mean_const = flatten(gps[1].mean)[1][1]
    s.noise = noise
    set_cache!(s, (posterior = s.cache.posterior, dirty = true))
    return s
end
