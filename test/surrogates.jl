using NonconvexBayesian, AbstractGPs, Test, LinearAlgebra
using IntervalArithmetic, ForwardDiff
using Zygote: Zygote
using ChainRulesCore: ChainRulesCore

@testitem "Surrogate" begin
    using NonconvexBayesian
    using Zygote: Zygote

    x = zeros(2)

    sf1 = NonconvexBayesian.ZeroOrderGPSurrogate(sum, x)
    lsf1 = NonconvexBayesian._lower_f(sf1)
    Zygote.gradient(lsf1, x)

    sf2 = NonconvexBayesian.ZeroOrderGPSurrogate(x -> x .^ 2, x)
    lsf2 = NonconvexBayesian._lower_f(sf2)
    Zygote.jacobian(lsf2, x)
end

@testitem "Surrogate gradients" begin
    using NonconvexBayesian, Test, LinearAlgebra
    using IntervalArithmetic, ForwardDiff
    using Zygote: Zygote

    h = 1e-6
    x0 = [1.234, 2.345]
    x = [1.0, 1.5]

    # Interval-mode lower-bound gradient matches finite differences.
    s1 = NonconvexBayesian.surrogate(x -> sqrt(x[2]), x0)
    s1.mode = :exact
    s1([1.5, 2.0])
    s1([0.8, 1.2])
    s1([2.0, 1.0])
    s1.mode = :interval
    f1 = z -> NonconvexBayesian.get_lo(s1(z))
    gfd = [
        (f1(x + h * [1.0, 0.0]) - f1(x - h * [1.0, 0.0])) / (2h),
        (f1(x + h * [0.0, 1.0]) - f1(x - h * [0.0, 1.0])) / (2h),
    ]
    gad = Zygote.gradient(f1, x)[1]
    @test gad ≈ gfd rtol = 1e-4

    # Broadcast lower-bound gradient (exercises the Dual handling in get_lo).
    s2 = NonconvexBayesian.surrogate(x -> [(2x[1])^3 - x[2], (-x[1] + 1)^3 - x[2]], x0)
    s2.mode = :exact
    s2([1.5, 2.0])
    s2([0.8, 1.2])
    s2([2.0, 1.0])
    s2.mode = :interval
    f2 = z -> sum(NonconvexBayesian.get_lo.(s2(z)))
    gfd2 = [
        (f2(x + h * [1.0, 0.0]) - f2(x - h * [1.0, 0.0])) / (2h),
        (f2(x + h * [0.0, 1.0]) - f2(x - h * [0.0, 1.0])) / (2h),
    ]
    gad2 = Zygote.gradient(f2, x)[1]
    @test gad2 ≈ gfd2 rtol = 1e-4

    # Exact-mode gradient is the gradient of the wrapped function.
    sf = NonconvexBayesian.ZeroOrderGPSurrogate(sum, [0.5, 1.5]; mode=:exact)
    g = Zygote.gradient(z -> NonconvexBayesian.get_lo(sf(z)), [0.5, 1.5])[1]
    @test g ≈ [1.0, 1.0]
    sf2 = NonconvexBayesian.ZeroOrderGPSurrogate(z -> z .^ 2, [0.5, 1.5]; mode=:exact)
    J = Zygote.jacobian(z -> NonconvexBayesian.get_lo(sf2(z)), [0.5, 1.5])[1]
    @test J ≈ diagm([1.0, 3.0])

    # Row-by-row Jacobian seeding (as done by NonconvexCore's Jacobian
    # assembly) through the equality-constraint transformation must not
    # descend into IntervalArithmetic's copysign/flipsign internals.
    seq = NonconvexBayesian.surrogate(z -> z[1] + z[2], x0)
    seq.mode = :exact
    seq([1.5, 2.0])
    seq([0.8, 1.2])
    seq.mode = :interval
    feq = z -> NonconvexBayesian._equality_f(seq)(z)
    Jeq = Zygote.jacobian(feq, x)[1]
    @test size(Jeq) == (2, 2)
    @test all(isfinite, Jeq)

    # get_lo re-types Zygote's interval-typed Dual seeds to scalar partials.
    d = ForwardDiff.Dual(interval(1.0, 2.0), (true,))
    r = NonconvexBayesian.get_lo(d)
    @test r isa ForwardDiff.Dual
    @test ForwardDiff.value(r) == 1.0
    @test ForwardDiff.partials(r, 1) ≈ 1.0
    d0 = ForwardDiff.Dual(interval(1.0, 2.0), (false,))
    r0 = NonconvexBayesian.get_lo(d0)
    @test ForwardDiff.value(r0) == 1.0
    @test ForwardDiff.partials(r0, 1) ≈ 0.0
end

@testitem "Surrogate cache and non-finite observations" begin
    using NonconvexBayesian, Test
    using IntervalArithmetic

    # The GP posterior must be cached across interval-mode evaluations: it is
    # only rebuilt when the data or the GP changes.
    s = NonconvexBayesian.surrogate(x -> x[1]^2 + x[2]^2, [1.234, 2.345])
    s([1.0, 1.0])
    post = s.cache.posterior
    s([1.0, 1.0])
    @test s.cache.posterior === post

    # Non-finite observations are skipped instead of poisoning the GP.
    s2 = NonconvexBayesian.ZeroOrderGPSurrogate(
        x -> 1 / (x[1] - 1.234), [2.0, 2.0]; mode=:exact
    )
    n = length(s2.X)
    r = @test_logs (:warn, r"non-finite") s2([1.234, 1.0])  # evaluates to Inf
    @test isnai(r)
    @test length(s2.X) == n
end

@testitem "Acquisition functions" begin
    using NonconvexBayesian, Test
    using IntervalArithmetic

    fq(x) = x[1]^2 + x[2]^2
    x0 = [1.0, 1.0]
    # EI and PI return degenerate intervals with non-positive values (the
    # acquisitions are negated since the sub-optimizer minimizes).
    for acq in (EI(), PI(), EI(0.1), PI(0.5))
        s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; acquisition=acq)
        s.mode = :exact
        s([1.1, 1.1])
        s([0.9, 0.9])
        s.mode = :interval
        v = s([1.0, 1.0])
        @test v isa Interval
        @test inf(v) <= 0
        @test inf(v) == sup(v) # degenerate
    end
    # EI is more negative far away from the data than at an observed point.
    s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; acquisition=EI())
    s.mode = :exact
    s([1.1, 1.1])
    s([0.9, 0.9])
    s.mode = :interval
    @test inf(s([5.0, 5.0])) < inf(s([1.0, 1.0])) <= 0
    # LCB keeps the [μ - ησ, μ + ησ] interval with η = std_multiple.
    s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; std_multiple=2.0)
    s.mode = :exact
    s([1.1, 1.1])
    s([0.9, 0.9])
    s.mode = :interval
    @test s.acquisition isa LCB
    @test s.acquisition.η == 2.0
    v = s([1.0, 1.0])
    @test inf(v) < sup(v)
    # GPUCB with a growing schedule becomes more conservative over time.
    s = NonconvexBayesian.ZeroOrderGPSurrogate(
        fq, x0; acquisition=GPUCB(1.0; schedule=t -> sqrt(2log(t + 1)))
    )
    s.mode = :exact
    s([1.1, 1.1])
    s([0.9, 0.9])
    s.mode = :interval
    v1 = s([2.0, 2.0])
    s.counter = 100
    v2 = s([2.0, 2.0])
    @test inf(v2) < inf(v1)  # larger β at a high-variance point lowers the bound
end

@testitem "First-order surrogate" begin
    using NonconvexBayesian, Test, LinearAlgebra
    using Zygote: Zygote
    using ForwardDiff
    using Statistics

    fq(x) = x[1]^2 + 3x[2]^2
    ∇fq(x) = [2x[1], 6x[2]]
    x0 = [0.3, 0.4]
    x = [0.5, 0.6]

    s = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source=∇fq)
    s.mode = :exact
    s([1.5, 2.0])
    s([-0.5, 1.0])
    s.mode = :interval

    # Gradient observations are recorded (one row per observation, one column
    # per input dimension).
    @test size(s.∇y[1], 2) == 2
    @test size(s.∇y[1], 1) >= 3

    # The rrule supplies the *predicted* gradient (from the gradient GPs) to
    # first-order sub-optimizers. This is an inexact-gradient scheme by
    # design, so it need not match finite differences of the surrogate value.
    f1 = z -> NonconvexBayesian.get_lo(s(z))
    gad = Zygote.gradient(f1, x)[1]
    @test all(isfinite, gad)

    # At a training point the gradient GPs interpolate the observed gradient.
    pg = s.cache.posterior[1]
    gtr = NonconvexBayesian._mean_gradient(s, pg, 1, [1.5, 2.0])
    @test gtr ≈ ∇fq([1.5, 2.0]) rtol = 1e-3

    # Exact-mode gradient equals the true gradient.
    s.mode = :exact
    ge = Zygote.gradient(z -> NonconvexBayesian.get_lo(s(z)), x)[1]
    @test ge ≈ ∇fq(x) rtol = 1e-6
    s.mode = :interval

    # grad_source = :none falls back to the AD gradient of the value GP mean.
    s0 = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source=:none)
    s0.mode = :exact
    s0([1.5, 2.0])
    s0.mode = :interval
    s0(x)  # build the posterior cache
    gm = NonconvexBayesian._mean_gradient(s0, s0.cache.posterior[1], 1, x)
    gm2 = ForwardDiff.gradient(z -> mean(s0.cache.posterior[1], [z])[1], x)
    @test gm ≈ gm2

    # ForwardDiff gradient observations.
    s1 = NonconvexBayesian.first_order_surrogate(fq, x0)  # grad_source = :forwarddiff
    s1.mode = :exact
    s1([1.5, 2.0])
    s1.mode = :interval
    s1(x)
    pg1 = s1.cache.posterior[1]
    gtr1 = NonconvexBayesian._mean_gradient(s1, pg1, 1, [1.5, 2.0])
    @test gtr1 ≈ ∇fq([1.5, 2.0]) rtol = 1e-3

    # Finite-difference gradient observations.
    s2 = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source=:finite)
    s2.mode = :exact
    s2([1.5, 2.0])
    s2.mode = :interval
    s2(x)
    pg2 = s2.cache.posterior[1]
    gtr2 = NonconvexBayesian._mean_gradient(s2, pg2, 1, [1.5, 2.0])
    @test gtr2 ≈ ∇fq([1.5, 2.0]) rtol = 1e-3
end

@testitem "EI and PI gradients" begin
    using NonconvexBayesian, Test, LinearAlgebra
    using Zygote: Zygote

    # The EI/PI rrule chains the acquisition gradient through the degenerate
    # interval, so it must match finite differences of the surrogate value
    # (unlike the first-order surrogate, whose gradient is a prediction).
    # Widely-spread training data keeps the posterior (and the acquisitions)
    # far from the underflow regime.
    fq(x) = x[1]^2 + x[2]^2
    x0 = [0.0, 0.0]
    x = [0.2, 0.0]  # near the best observed point: substantial EI/PI
    h = 1e-6
    for acq in (EI(), PI(0.1))
        s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; acquisition=acq)
        s.mode = :exact
        s([1.0, 1.0])
        s([2.0, 0.0])
        s([0.0, 2.0])
        s.mode = :interval
        f1 = z -> NonconvexBayesian.get_lo(s(z))
        @test abs(f1(x)) > 1e-5  # the acquisition is not degenerate here
        gfd = [
            (f1(x + h * [1.0, 0.0]) - f1(x - h * [1.0, 0.0])) / (2h),
            (f1(x + h * [0.0, 1.0]) - f1(x - h * [0.0, 1.0])) / (2h),
        ]
        gad = Zygote.gradient(f1, x)[1]
        @test all(isfinite, gad)
        @test gad ≈ gfd rtol = 1e-4 atol = 1e-8
    end
end

@testitem "First-order surrogate, multi-output" begin
    using NonconvexBayesian, Test, LinearAlgebra
    using Zygote: Zygote

    fq2(x) = [x[1]^2 + 3x[2]^2, 2x[1] - x[2]]
    x0 = [0.3, 0.4]
    s = NonconvexBayesian.first_order_surrogate(fq2, x0)  # ForwardDiff gradients
    s.mode = :exact
    s([1.5, 2.0])
    s([-0.5, 1.0])
    s.mode = :interval
    s([0.5, 0.6])  # build the posterior cache

    @test s.N == 2
    @test size(s.∇y[1], 2) == 2
    @test size(s.∇y[2], 2) == 2
    @test size(s.∇y[1], 1) >= 3
    @test length(s.∇gps) == 2 && length(s.∇gps[1]) == 2

    # The gradient GPs interpolate each output's observed gradients.
    pg1 = s.cache.posterior[1]
    pg2 = s.cache.posterior[2]
    @test NonconvexBayesian._mean_gradient(s, pg1, 1, [1.5, 2.0]) ≈ [3.0, 12.0] rtol = 1e-3
    @test NonconvexBayesian._mean_gradient(s, pg2, 2, [1.5, 2.0]) ≈ [2.0, -1.0] rtol = 1e-3

    # The multi-output rrule runs through Zygote and stays finite.
    fsum = z -> sum(NonconvexBayesian.get_lo.(s(z)))
    gad = Zygote.gradient(fsum, [0.5, 0.6])[1]
    @test all(isfinite, gad)
end

@testitem "Gradient GP hyperparameter refitting" begin
    using NonconvexBayesian, Test, LinearAlgebra

    fq(x) = x[1]^2 + 3x[2]^2
    ∇fq(x) = [2x[1], 6x[2]]
    x0 = [0.3, 0.4]
    # every = 1, skip = 0 forces a refit (value + gradient GPs) per evaluation.
    s = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source=∇fq, every=1, skip=0)
    s.mode = :exact
    s([1.5, 2.0])
    s([-0.5, 1.0])
    @test s.counter == 2  # the factory no longer double-evaluates x0
    s.mode = :interval
    s([0.5, 0.6])
    # After the refits, the gradient GPs still interpolate at training points.
    gtr = NonconvexBayesian._mean_gradient(s, s.cache.posterior[1], 1, [1.5, 2.0])
    @test gtr ≈ ∇fq([1.5, 2.0]) rtol = 1e-3
end

@testitem "fit_noise" begin
    using NonconvexBayesian, Test

    fq(x) = x[1]^2 + x[2]^2
    s = NonconvexBayesian.surrogate(fq, [1.0, 1.0]; fit_noise=true, every=1, skip=0)
    s.mode = :exact
    s([1.1, 1.1])
    # The MLE included the noise, which stays within its [1e-6, Inf] bounds
    # (the floor is above Ipopt's default bound relaxation).
    @test s.noise >= 1e-7
    @test isfinite(s.noise)
end

@testitem "Deduplication helper" begin
    using NonconvexBayesian, Test

    fq(x) = x[1]^2 + x[2]^2
    s = NonconvexBayesian.surrogate(fq, [1.0, 1.0])
    s.mode = :exact
    s([1.1, 1.1])
    s.mode = :interval
    @test NonconvexBayesian._mindist([s], [1.1, 1.1]) < 1e-8
    @test NonconvexBayesian._mindist([s], [5.0, 5.0]) > 1.0
end

@testitem "Defensive paths" begin
    using NonconvexBayesian, Test, LinearAlgebra
    using IntervalArithmetic
    using ChainRulesCore: ChainRulesCore

    # LCB default constructor.
    @test LCB().η == 3.0

    # The gradient-recording trait (via invokelatest so the one-liner
    # methods are exercised through runtime dispatch).
    @test !Base.invokelatest(
        NonconvexBayesian._records_gradients,
        NonconvexBayesian.ZeroOrderGPSurrogate(x -> x[1], [0.0]),
    )
    @test Base.invokelatest(
        NonconvexBayesian._records_gradients,
        NonconvexBayesian.FirstOrderGPSurrogate(x -> x[1], [0.0]; grad_source=:none),
    )
    @test Base.invokelatest(
        NonconvexBayesian._records_gradients,
        NonconvexBayesian.CoKrigingSurrogate(x -> x[1], [0.0]; grad_source=:none),
    )

    # get_lo on a Real is the identity.
    @test NonconvexBayesian.get_lo(3.5) == 3.5

    # The interval perturbation helpers handle values, intervals and zero
    # cotangents directly.
    @test NonconvexBayesian._interval_perturbation(interval(1.0, 2.0)) == (1.0, 2.0)
    @test NonconvexBayesian._interval_perturbation(3.0) == (3.0, 0.0)
    @test NonconvexBayesian._interval_perturbation(ChainRulesCore.ZeroTangent()) ==
        (0.0, 0.0)
    @test NonconvexBayesian._perturbation_sum(interval(1.0, 2.0)) == 3.0

    # ScaledSobolIterator length.
    @test length(NonconvexBayesian.ScaledSobolIterator([0.0], [1.0], 5)) == 5

    # The owned get_lo and _neg_interval ChainRules rules (structural
    # tangents; zero and non-real cotangent branches).
    b = bareinterval(interval(1.0, 2.0))
    yb, pbb = ChainRulesCore.rrule(NonconvexBayesian.get_lo, interval(1.0, 2.0))
    @test yb == 1.0
    tb = pbb(1.0)[2]
    @test tb isa ChainRulesCore.Tangent
    @test tb.bareinterval.lo == 1.0 && tb.bareinterval.hi == 0.0
    @test pbb(ChainRulesCore.ZeroTangent())[2] isa ChainRulesCore.ZeroTangent
    @test pbb(nothing)[2] isa ChainRulesCore.Tangent  # non-Real seed falls back to zero
    yn, pbn = ChainRulesCore.rrule(NonconvexBayesian._neg_interval, interval(1.0, 2.0))
    @test inf(yn) == -2.0 && sup(yn) == -1.0
    tn = pbn(1.0)[2]  # seeding the output lo by 1 negates into the input hi
    @test tn.bareinterval.lo == 0.0 && tn.bareinterval.hi == -1.0
    @test pbn(ChainRulesCore.ZeroTangent())[2] isa ChainRulesCore.ZeroTangent
end

@testitem "EI/PI near-zero variance branches" begin
    using NonconvexBayesian, Test
    using AbstractGPs

    # A noise-free posterior has σ = 0 at the observed points, which forces
    # the σ < eps branches of the EI/PI acquisitions.
    fq(x) = x[1]^2 + x[2]^2
    g0 = GP(ConstMean(0.0), SqExponentialKernel())
    X0 = [[1.0, 1.0], [2.0, 0.0]]
    y0 = [1.0, 2.0]
    pg0 = posterior(g0(X0, 0.0), y0)
    xobs = [1.0, 1.0]
    @test abs(NonconvexBayesian._acq_value(EI(), pg0, X0, xobs, 0)) == 0.0
    @test abs(NonconvexBayesian._acq_value(PI(0.1), pg0, X0, xobs, 0)) == 0.0
end

@testitem "fit_mle! rrule" begin
    using NonconvexBayesian, Test
    using ChainRulesCore: ChainRulesCore

    fq(x) = x[1]^2 + x[2]^2
    s = NonconvexBayesian.surrogate(fq, [1.0, 1.0]; fit_prior=false)
    y, pb = ChainRulesCore.rrule(
        NonconvexBayesian.fit_mle!, s, s.gps, s.X, s.y, s.noise, s.last, s.fit_noise
    )
    @test y === s
    cot = pb(1.0)
    @test cot isa NTuple{8,ChainRulesCore.NoTangent}
end

@testitem "Non-finite gradient observations" begin
    using NonconvexBayesian, Test

    # The constructor records the finite gradient at x0; the non-finite
    # gradient at a later observation is skipped with a warning.
    s = NonconvexBayesian.first_order_surrogate(
        x -> x[1] + x[2], [0.0, 0.0]; grad_source=x -> x[1] > 0.5 ? [Inf, 1.0] : [1.0, 1.0]
    )
    nrows = size(s.∇y[1], 1)
    s.mode = :exact
    @test_logs (:warn, r"gradient") s([1.0, 1.0])
    @test size(s.∇y[1], 1) == nrows  # the non-finite gradient was skipped
end
