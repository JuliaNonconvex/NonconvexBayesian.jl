using NonconvexBayesian, NonconvexIpopt, AbstractGPs
using LinearAlgebra, Test
using IntervalArithmetic, ForwardDiff
using NonconvexCore: AbstractOptimizer, ConstraintCapabilities, solver_capabilities, tovecmodel
import Zygote

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

# A sub-optimizer that declares it does not support semidefinite constraints,
# used to test capability forwarding through BayesOptAlg.
struct NoSDOptimizer <: AbstractOptimizer end
NonconvexCore.solver_capabilities(::NoSDOptimizer) = ConstraintCapabilities(sd = false)

@testset "Surrogate" begin
    x = zeros(2)

    sf1 = NonconvexBayesian.ZeroOrderGPSurrogate(sum, x)
    lsf1 = NonconvexBayesian._lower_f(sf1)
    Zygote.gradient(lsf1, x)

    sf2 = NonconvexBayesian.ZeroOrderGPSurrogate(x -> x .^ 2, x)
    lsf2 = NonconvexBayesian._lower_f(sf2)
    Zygote.jacobian(lsf2, x)
end

@testset "Surrogate gradients" begin
    h = 1e-6
    x0 = [1.234, 2.345]
    x = [1.0, 1.5]

    # Interval-mode lower-bound gradient matches finite differences.
    s1 = NonconvexBayesian.surrogate(x -> sqrt(x[2]), x0)
    s1.mode = :exact
    s1([1.5, 2.0]); s1([0.8, 1.2]); s1([2.0, 1.0])
    s1.mode = :interval
    f1 = z -> NonconvexBayesian.get_lo(s1(z))
    gfd = [(f1(x + h * [1.0, 0.0]) - f1(x - h * [1.0, 0.0])) / (2h),
        (f1(x + h * [0.0, 1.0]) - f1(x - h * [0.0, 1.0])) / (2h)]
    gad = Zygote.gradient(f1, x)[1]
    @test gad ≈ gfd rtol = 1e-4

    # Broadcast lower-bound gradient (exercises the Dual handling in get_lo).
    s2 = NonconvexBayesian.surrogate(x -> [(2x[1])^3 - x[2], (-x[1] + 1)^3 - x[2]], x0)
    s2.mode = :exact
    s2([1.5, 2.0]); s2([0.8, 1.2]); s2([2.0, 1.0])
    s2.mode = :interval
    f2 = z -> sum(NonconvexBayesian.get_lo.(s2(z)))
    gfd2 = [(f2(x + h * [1.0, 0.0]) - f2(x - h * [1.0, 0.0])) / (2h),
        (f2(x + h * [0.0, 1.0]) - f2(x - h * [0.0, 1.0])) / (2h)]
    gad2 = Zygote.gradient(f2, x)[1]
    @test gad2 ≈ gfd2 rtol = 1e-4

    # Exact-mode gradient is the gradient of the wrapped function.
    sf = NonconvexBayesian.ZeroOrderGPSurrogate(sum, [0.5, 1.5]; mode = :exact)
    g = Zygote.gradient(z -> NonconvexBayesian.get_lo(sf(z)), [0.5, 1.5])[1]
    @test g ≈ [1.0, 1.0]
    sf2 = NonconvexBayesian.ZeroOrderGPSurrogate(z -> z .^ 2, [0.5, 1.5]; mode = :exact)
    J = Zygote.jacobian(z -> NonconvexBayesian.get_lo(sf2(z)), [0.5, 1.5])[1]
    @test J ≈ diagm([1.0, 3.0])

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

@testset "Surrogate cache and non-finite observations" begin
    # The GP posterior must be cached across interval-mode evaluations: it is
    # only rebuilt when the data or the GP changes.
    s = NonconvexBayesian.surrogate(x -> x[1]^2 + x[2]^2, [1.234, 2.345])
    s([1.0, 1.0])
    post = s.cache.posterior
    s([1.0, 1.0])
    @test s.cache.posterior === post

    # Non-finite observations are skipped instead of poisoning the GP.
    s2 = NonconvexBayesian.ZeroOrderGPSurrogate(x -> 1 / (x[1] - 1.234), [2.0, 2.0]; mode = :exact)
    n = length(s2.X)
    r = @test_logs (:warn, r"non-finite") s2([1.234, 1.0])  # evaluates to Inf
    @test isnai(r)
    @test length(s2.X) == n
end

@testset "Acquisition functions" begin
    fq(x) = x[1]^2 + x[2]^2
    x0 = [1.0, 1.0]
    # EI and PI return degenerate intervals with non-positive values (the
    # acquisitions are negated since the sub-optimizer minimizes).
    for acq in (EI(), PI(), EI(0.1), PI(0.5))
        s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; acquisition = acq)
        s.mode = :exact
        s([1.1, 1.1]); s([0.9, 0.9])
        s.mode = :interval
        v = s([1.0, 1.0])
        @test v isa Interval
        @test inf(v) <= 0
        @test inf(v) == sup(v) # degenerate
    end
    # EI is more negative far away from the data than at an observed point.
    s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; acquisition = EI())
    s.mode = :exact
    s([1.1, 1.1]); s([0.9, 0.9])
    s.mode = :interval
    @test inf(s([5.0, 5.0])) < inf(s([1.0, 1.0])) <= 0
    # LCB keeps the [μ - ησ, μ + ησ] interval with η = std_multiple.
    s = NonconvexBayesian.ZeroOrderGPSurrogate(fq, x0; std_multiple = 2.0)
    s.mode = :exact
    s([1.1, 1.1]); s([0.9, 0.9])
    s.mode = :interval
    @test s.acquisition isa LCB
    @test s.acquisition.η == 2.0
    v = s([1.0, 1.0])
    @test inf(v) < sup(v)
    # GPUCB with a growing schedule becomes more conservative over time.
    s = NonconvexBayesian.ZeroOrderGPSurrogate(
        fq,
        x0;
        acquisition = GPUCB(1.0; schedule = t -> sqrt(2log(t + 1))),
    )
    s.mode = :exact
    s([1.1, 1.1]); s([0.9, 0.9])
    s.mode = :interval
    v1 = s([2.0, 2.0])
    s.counter = 100
    v2 = s([2.0, 2.0])
    @test inf(v2) < inf(v1)  # larger β at a high-variance point lowers the bound
end

@testset "First-order surrogate" begin
    fq(x) = x[1]^2 + 3x[2]^2
    ∇fq(x) = [2x[1], 6x[2]]
    x0 = [0.3, 0.4]
    x = [0.5, 0.6]

    s = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source = ∇fq)
    s.mode = :exact
    s([1.5, 2.0]); s([-0.5, 1.0])
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
    s0 = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source = :none)
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
    s2 = NonconvexBayesian.first_order_surrogate(fq, x0; grad_source = :finite)
    s2.mode = :exact
    s2([1.5, 2.0])
    s2.mode = :interval
    s2(x)
    pg2 = s2.cache.posterior[1]
    gtr2 = NonconvexBayesian._mean_gradient(s2, pg2, 1, [1.5, 2.0])
    @test gtr2 ≈ ∇fq([1.5, 2.0]) rtol = 1e-3
end

@testset "Cheap objective and constraints" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 10,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Expensive objective" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 10,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            skip = 2,
            every = 2,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Expensive objective - postoptimize" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 10,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = true,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-5
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-5
    end
end

@testset "Expensive objective - no preinitialize" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 10,
            ctol = 1e-4,
            initialize = false,
            postoptimize = false,
            skip = 2,
            every = 2,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-5
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-5
    end
end

@testset "Expensive inequality constraint" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:expensive])
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 100,
            ctol = 1e-4,
            ninit = 5,
            initialize = true,
            postoptimize = false,
            last = 20,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Expensive inequality constraint with equality" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:expensive])
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 20,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            last = 10,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Two expensive inequality constraint" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:expensive])
        add_ineq_constraint!(m, x -> g(x, -1, 1), flags = [:expensive])

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 40,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            every = 5,
            skip = 2,
            last = 10,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Expensive equality constraint" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27, flags = [:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 30,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            last = 10,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Expensive block constraint" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> [g(x, 2, 0), g(x, -1, 1)], flags = [:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 40,
            ctol = 1e-4,
            ninit = 5,
            initialize = true,
            postoptimize = false,
            skip = 2,
            every = 2,
            last = 10,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "All expensive no equality" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> [g(x, 2, 0), g(x, -1, 1)], flags = [:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(),
            maxiter = 50,
            ctol = 1e-3,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            last = 10,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "All expensive with equality" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> [g(x, 2, 0), g(x, -1, 1)], flags = [:expensive])
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27, flags = [:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 100,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            last = 30,
            skip = 2,
            every = 2,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Passing surrogates in" begin
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        x0 = [1.234, 2.345]
        s1 = NonconvexBayesian.surrogate(f, x0)
        s2 = NonconvexBayesian.surrogate(x -> [g(x, 2, 0), g(x, -1, 1)], x0)

        m = Model()
        set_objective!(m, x -> NonconvexBayesian.get_lo(s1(x)), flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> NonconvexBayesian.get_lo.(s2(x)), flags = [:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 50,
            ctol = 1e-4,
            ninit = 2,
            initialize = true,
            postoptimize = false,
            fit_prior = fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options, surrogates = [s1, s2])
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Init strategies" begin
    @testset "init_strategy = $init_strategy" for init_strategy in (:sobol_direct, :acquisition)
        m = Model()
        set_objective!(m, f, flags = [:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 8,
            ctol = 1e-4,
            ninit = 3,
            init_strategy = init_strategy,
            postoptimize = false,
            fit_prior = false,
        )
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testset "Acquisition multistart and deduplication" begin
    m = Model()
    set_objective!(m, f, flags = [:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options = IpoptOptions(print_level = 0),
        maxiter = 25,
        ctol = 1e-4,
        ninit = 3,
        postoptimize = false,
        fit_prior = false,
        acq_multistart = 3,
        deduplicate = true,
        xtol_dup = 1e-6,
    )
    r = optimize(m, alg, [1.234, 2.345], options = options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
end

@testset "First-order Bayesian optimization" begin
    m = Model()
    set_objective!(m, f, flags = [:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options = IpoptOptions(print_level = 0),
        maxiter = 10,
        ctol = 1e-4,
        ninit = 2,
        postoptimize = false,
        fit_prior = false,
        first_order = true,
        grad_source = :forwarddiff,
    )
    r = optimize(m, alg, [1.234, 2.345], options = options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
end

@testset "Protocol integration" begin
    ncalls = Ref(0)
    fcounted(x) = (ncalls[] += 1; f(x))
    m = Model()
    set_objective!(m, fcounted, flags = [:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    cb_counter = Ref(0)
    options = BayesOptOptions(
        sub_options = IpoptOptions(print_level = 0),
        maxiter = 5,
        ctol = 1e-4,
        ninit = 2,
        postoptimize = false,
        fit_prior = false,
        keep_all = true,
    )
    r = optimize(
        m,
        alg,
        [1.234, 2.345],
        options = options,
        callback = sol -> (cb_counter[] += 1),
    )
    @test cb_counter[] == 5            # callback once per BO iteration
    @test r.iter == 5                  # exactly maxiter iterations (off-by-one fixed)
    @test r.fcalls >= 5                # expensive evaluations performed
    @test r.initial_x == [1.234, 2.345]
    @test r.tol isa NonconvexCore.Tolerance
    @test r.convstate isa NonconvexCore.ConvergenceState
    @test r.maxiter_reached == !r.convstate.converged
    @test r.trajectory !== nothing
    @test length(r.trajectory) == 5
    @test r.optimizer === alg

    # A short run without convergence reports maxiter_reached = true.
    r2 = optimize(
        m,
        alg,
        [1.234, 2.345],
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 2,
            ctol = 1e-12,
            ftol = 1e-12,
            initialize = false,
            postoptimize = false,
            fit_prior = false,
        ),
    )
    @test r2.iter == 2
    @test r2.maxiter_reached

    # A long run converges before maxiter.
    r3 = optimize(
        m,
        alg,
        [1.234, 2.345],
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 200,
            ctol = 1e-4,
            ftol = 1e-4,
            ninit = 2,
            postoptimize = false,
            fit_prior = false,
        ),
    )
    @test !r3.maxiter_reached
    @test r3.convstate.converged

    # verbose mode runs without error.
    rv = optimize(
        m,
        alg,
        [1.234, 2.345],
        options = BayesOptOptions(
            sub_options = IpoptOptions(print_level = 0),
            maxiter = 3,
            ninit = 1,
            postoptimize = false,
            fit_prior = false,
            verbose = true,
        ),
    )
    @test rv.iter == 3
end

@testset "reset!" begin
    m = Model()
    set_objective!(m, f, flags = [:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options = IpoptOptions(print_level = 0),
        maxiter = 4,
        ctol = 1e-4,
        ftol = 1e-12,   # never converge early: deterministic evaluation counts
        ninit = 2,
        postoptimize = false,
        fit_prior = false,
    )
    vm, x0, _ = tovecmodel(m, [1.234, 2.345])
    w = Workspace(vm, alg, x0; options = options)
    r1 = NonconvexCore.optimize!(w)
    fc1 = r1.fcalls
    @test fc1 > 0
    nX1 = length(w.surrogates[1].X)
    # Re-running without a reset keeps accumulating surrogate data.
    r_accum = NonconvexCore.optimize!(w)
    @test r_accum.fcalls == r1.fcalls
    @test length(w.surrogates[1].X) > nX1
    # After reset!, the surrogates are rebuilt and the data restarts.
    NonconvexCore.reset!(w, [1.234, 2.345])
    r2 = NonconvexCore.optimize!(w)
    @test r2.fcalls == fc1
    @test length(w.surrogates[1].X) <= nX1
end

@testset "Capabilities" begin
    caps = solver_capabilities(BayesOptAlg(IpoptAlg()))
    @test caps.eq === true
    @test caps.ineq === true
    @test caps.box === true
    @test caps.sd === false           # forwarded from IpoptAlg's declaration
    @test caps.integer === false      # forwarded from IpoptAlg's declaration

    # sd is forwarded from the sub-solver.
    caps2 = solver_capabilities(BayesOptAlg(NoSDOptimizer()))
    @test caps2.sd === false

    # A model with an sd constraint errors at the generic optimize level.
    m = Model()
    set_objective!(m, f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_sd_constraint!(m, x -> Matrix{Float64}(I, 2, 2))
    err = try
        optimize(
            m,
            BayesOptAlg(NoSDOptimizer()),
            [1.234, 2.345],
            options = BayesOptOptions(sub_options = nothing),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("semidefinite", sprint(showerror, err))
end

@testset "MLE controls" begin
    # The extension provides the default hyperparameter optimizer (Ipopt).
    @test NonconvexBayesian._fit_algorithm(NonconvexBayesian.NoFitOptimizer(), 5) isa Tuple

    # fit_max_iter, fit_bounds and fit_optimizer are accepted and refits run.
    s = NonconvexBayesian.surrogate(
        f,
        [1.234, 2.345];
        every = 1,
        skip = 0,
        fit_max_iter = 3,
        fit_bounds = (lb = -10.0, ub = 10.0),
        fit_optimizer = (IpoptAlg(), IpoptOptions(max_iter = 3, print_level = 0)),
    )
    s.mode = :exact
    s([1.5, 2.0])
    @test s.counter == 2  # one from the surrogate factory, one from this call
    # Invalid fit_bounds lengths are rejected.
    @test_throws ArgumentError NonconvexBayesian._expand_to([1.0, 2.0], 3)
end
