using NonconvexBayesian, AbstractGPs, Test, LinearAlgebra
using IntervalArithmetic, ForwardDiff
using Zygote: Zygote

@testitem "Co-kriging surrogate" begin
    using NonconvexBayesian, AbstractGPs, Test, LinearAlgebra
    using IntervalArithmetic, ForwardDiff
    using Zygote: Zygote

    fq(x) = x[1]^2 + 3x[2]^2
    ∇fq(x) = [2x[1], 6x[2]]
    x0 = [0.3, 0.4]
    x = [0.5, 0.6]
    h = 1e-6

    s = NonconvexBayesian.co_kriging_surrogate(fq, x0; grad_source=∇fq)
    s.mode = :exact
    s([1.5, 2.0])
    s([-0.5, 1.0])
    s.mode = :interval
    s(x)  # build the cache

    # Gradient observations recorded.
    @test size(s.∇y[1], 2) == 2
    @test size(s.∇y[1], 1) >= 3

    # The posterior is the joint value+gradient model.
    pg = s.cache.posterior[1]
    @test pg isa NonconvexBayesian.CoKrigingPosterior

    # Value predictions are finite; variance is (numerically) non-negative.
    @test all(isfinite, AbstractGPs.mean(pg, [x]))
    @test AbstractGPs.var(pg, [x])[1] > -1e-8
    @test AbstractGPs.var(pg, x) > -1e-8  # single-point form

    # Self-consistency: the rrule gradient is the exact derivative of the
    # predicted value, so it matches finite differences (unlike the
    # independent first-order surrogate).
    f1 = z -> NonconvexBayesian.get_lo(s(z))
    gfd = [
        (f1(x + h * [1.0, 0.0]) - f1(x - h * [1.0, 0.0])) / (2h),
        (f1(x + h * [0.0, 1.0]) - f1(x - h * [0.0, 1.0])) / (2h),
    ]
    gad = Zygote.gradient(f1, x)[1]
    @test gad ≈ gfd rtol = 1e-4

    # Gradient-enhanced property: at a training point the value mean's
    # derivative interpolates the observed gradient...
    gtr = ForwardDiff.gradient(z -> AbstractGPs.mean(pg, z), [1.5, 2.0])
    @test gtr ≈ ∇fq([1.5, 2.0]) rtol = 1e-3
    # ... and the surrogate's predicted gradient equals that derivative.
    gm = NonconvexBayesian._mean_gradient(s, pg, 1, [1.5, 2.0])
    @test gm ≈ gtr

    # Value interpolation at a training point.
    @test AbstractGPs.mean(pg, [1.5, 2.0])[1] ≈ fq([1.5, 2.0]) rtol = 1e-3

    # Exact-mode gradient equals the true gradient.
    s.mode = :exact
    ge = Zygote.gradient(z -> NonconvexBayesian.get_lo(s(z)), x)[1]
    @test ge ≈ ∇fq(x) rtol = 1e-6
    s.mode = :interval

    # EI acquisition through the co-kriging posterior.
    sEI = NonconvexBayesian.co_kriging_surrogate(fq, x0; grad_source=∇fq, acquisition=EI())
    sEI.mode = :exact
    sEI([1.5, 2.0])
    sEI([-0.5, 1.0])
    sEI.mode = :interval
    v = sEI(x)
    @test v isa Interval && inf(v) <= sup(v)

    # The posterior constructor validates the gradient-data shape.
    @test_throws ErrorException NonconvexBayesian.CoKrigingPosterior(
        SqExponentialKernel(),
        0.0,
        [[0.3, 0.4], [1.5, 2.0]],
        [1.0, 2.0],
        zeros(3, 2),  # wrong number of rows
        1e-8,
        1e-8,
    )
end

@testitem "Co-kriging MLE" begin
    using NonconvexBayesian, AbstractGPs, Test, LinearAlgebra
    using ForwardDiff

    fq(x) = x[1]^2 + 3x[2]^2
    ∇fq(x) = [2x[1], 6x[2]]
    # every = 1, skip = 0 forces a joint value+gradient MLE refit per
    # evaluation (with fit_noise to exercise the noise-fitting path).
    s = NonconvexBayesian.co_kriging_surrogate(
        fq, [0.3, 0.4]; grad_source=∇fq, every=1, skip=0, fit_noise=true
    )
    s.mode = :exact
    s([1.5, 2.0])
    s([-0.5, 1.0])
    @test s.counter == 2  # the factory no longer double-evaluates x0
    @test s.noise >= 1e-7
    s.mode = :interval
    @test isfinite(NonconvexBayesian.get_lo(s([0.5, 0.6])))
    # Interpolation still holds after the joint-MLE refits.
    pg = s.cache.posterior[1]
    gtr = ForwardDiff.gradient(z -> AbstractGPs.mean(pg, z), [1.5, 2.0])
    @test gtr ≈ ∇fq([1.5, 2.0]) rtol = 1e-3

    # The `last` window is applied to the values and gradients consistently.
    s2 = NonconvexBayesian.co_kriging_surrogate(fq, [0.3, 0.4]; grad_source=∇fq, last=2)
    s2.mode = :exact
    s2([1.5, 2.0])
    s2([-0.5, 1.0])
    s2([0.8, 0.9])
    Xw, yw, ∇w = NonconvexBayesian._window_cokriging_data(s2, s2.last)
    @test length(Xw) == 2
    @test length(yw[1]) == 2
    @test size(∇w[1], 1) == 2
    # `last = :all` keeps everything.
    Xa, ya, ∇a = NonconvexBayesian._window_cokriging_data(s2, :all)
    @test length(Xa) == 4
    @test length(ya[1]) == 4
    @test size(∇a[1], 1) == 4
end

@testitem "Co-kriging multi-output" begin
    using NonconvexBayesian, AbstractGPs, Test, LinearAlgebra
    using ForwardDiff
    using Zygote: Zygote

    fq2(x) = [x[1]^2 + 3x[2]^2, 2x[1] - x[2]]
    s = NonconvexBayesian.co_kriging_surrogate(fq2, [0.3, 0.4])
    s.mode = :exact
    s([1.5, 2.0])
    s([-0.5, 1.0])
    s.mode = :interval
    s([0.5, 0.6])
    @test s.N == 2
    @test length(s.cache.posterior) == 2
    gtr2 = ForwardDiff.gradient(z -> AbstractGPs.mean(s.cache.posterior[2], z), [1.5, 2.0])
    @test gtr2 ≈ [2.0, -1.0] rtol = 1e-3
    fsum = z -> sum(NonconvexBayesian.get_lo.(s(z)))
    @test all(isfinite, Zygote.gradient(fsum, [0.5, 0.6])[1])
end

@testitem "Co-kriging Bayesian optimization" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    # Rosenbrock: a smooth benchmark where the gradient-enhanced surrogate
    # converges to the global optimum.
    ros(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    m = Model()
    set_objective!(m, ros, flags=[:expensive])
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=30,
        ctol=1e-4,
        ninit=5,
        postoptimize=false,
        fit_prior=false,
        co_kriging=true,
        grad_source=:forwarddiff,
        acq_multistart=3,
    )
    r = optimize(m, alg, [-1.5, 1.0], options=options)
    @test r.minimum < 1e-2
    @test norm(r.minimizer - [1.0, 1.0]) < 5e-2

    # A constrained problem with co-kriging surrogates and postoptimize
    # polish on the exact model.
    m2 = Model()
    set_objective!(m2, f, flags=[:expensive])
    addvar!(m2, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m2, x -> g(x, 2, 0), flags=[:expensive])
    add_ineq_constraint!(m2, x -> g(x, -1, 1))
    options2 = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=10,
        ctol=1e-4,
        ninit=3,
        postoptimize=true,
        fit_prior=false,
        co_kriging=true,
        grad_source=:forwarddiff,
    )
    r2 = optimize(m2, alg, [1.234, 2.345], options=options2)
    @test abs(r2.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r2.minimizer - [1 / 3, 8 / 27]) < 1e-3
end
