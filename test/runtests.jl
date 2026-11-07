using NonconvexBayesian, NonconvexIpopt, NonconvexNLopt
using LinearAlgebra, Test
using IntervalArithmetic, ForwardDiff
import Zygote

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

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
            maxiter = 20,
            ctol = 1e-4,
            ninit = 2,
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
