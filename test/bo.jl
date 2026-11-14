using NonconvexBayesian, NonconvexIpopt, AbstractGPs
using LinearAlgebra, Test
using IntervalArithmetic, ForwardDiff
using NonconvexCore:
    AbstractOptimizer, ConstraintCapabilities, solver_capabilities, tovecmodel
using Zygote: Zygote
using ChainRulesCore: ChainRulesCore

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

# A sub-optimizer that declares it does not support semidefinite constraints,
# used to test capability forwarding through BayesOptAlg.
struct NoSDOptimizer <: AbstractOptimizer end
function NonconvexCore.solver_capabilities(::NoSDOptimizer)
    return ConstraintCapabilities(; linear_sd=false, nonlinear_sd=false)
end

@testitem "Cheap objective and constraints" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=10,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Expensive objective" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=10,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            skip=2,
            every=2,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Expensive objective - postoptimize" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=10,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=true,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-5
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-5
    end
end

@testitem "Expensive objective - no preinitialize" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=10,
            ctol=1e-4,
            initialize=false,
            postoptimize=false,
            skip=2,
            every=2,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-5
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-5
    end
end

@testitem "Expensive inequality constraint" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:expensive])
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=100,
            ctol=1e-4,
            ninit=5,
            initialize=true,
            postoptimize=false,
            last=20,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Expensive inequality constraint with equality" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:expensive])
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=20,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            last=10,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Two expensive inequality constraint" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:expensive])
        add_ineq_constraint!(m, x -> g(x, -1, 1), flags=[:expensive])

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=40,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            every=5,
            skip=2,
            last=10,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Expensive equality constraint" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27, flags=[:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=30,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            last=10,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Expensive block constraint" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    # The hyperparameter MLE is nonconvex, so its basin can differ across
    # machines; postoptimize polishes the incumbent on the exact model so
    # the assertions do not depend on the MLE trajectory.
    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> [g(x, 2, 0), g(x, -1, 1)], flags=[:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=40,
            ctol=1e-4,
            ninit=5,
            initialize=true,
            postoptimize=true,
            skip=2,
            every=2,
            last=10,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "All expensive no equality" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> [g(x, 2, 0), g(x, -1, 1)], flags=[:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(),
            maxiter=50,
            ctol=1e-3,
            ninit=2,
            initialize=true,
            postoptimize=false,
            last=10,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "All expensive with equality" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> [g(x, 2, 0), g(x, -1, 1)], flags=[:expensive])
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27, flags=[:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=100,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            last=30,
            skip=2,
            every=2,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Passing surrogates in" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Fit prior: $fit_prior" for fit_prior in (true, false)
        x0 = [1.234, 2.345]
        s1 = NonconvexBayesian.surrogate(f, x0)
        s2 = NonconvexBayesian.surrogate(x -> [g(x, 2, 0), g(x, -1, 1)], x0)

        m = Model()
        set_objective!(m, x -> NonconvexBayesian.get_lo(s1(x)), flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> NonconvexBayesian.get_lo.(s2(x)), flags=[:expensive])
        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=50,
            ctol=1e-4,
            ninit=2,
            initialize=true,
            postoptimize=false,
            fit_prior=fit_prior,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options, surrogates=[s1, s2])
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Init strategies" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "init_strategy = $init_strategy" for init_strategy in
                                                  (:sobol_direct, :acquisition)
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = BayesOptAlg(IpoptAlg())
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=8,
            ctol=1e-4,
            ninit=3,
            init_strategy=init_strategy,
            postoptimize=false,
            fit_prior=false,
        )
        r = optimize(m, alg, [1.234, 2.345], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
    end
end

@testitem "Acquisition multistart and deduplication" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=25,
        ctol=1e-4,
        ninit=3,
        postoptimize=false,
        fit_prior=false,
        acq_multistart=3,
        deduplicate=true,
        xtol_dup=1e-6,
    )
    r = optimize(m, alg, [1.234, 2.345], options=options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
end

@testitem "First-order Bayesian optimization" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=10,
        ctol=1e-4,
        ninit=2,
        postoptimize=false,
        fit_prior=false,
        first_order=true,
        grad_source=:forwarddiff,
    )
    r = optimize(m, alg, [1.234, 2.345], options=options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
end

@testitem "Protocol integration" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra
    using NonconvexCore

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    ncalls = Ref(0)
    fcounted(x) = (ncalls[] += 1; f(x))
    m = Model()
    set_objective!(m, fcounted, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    cb_counter = Ref(0)
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=5,
        ctol=1e-4,
        ninit=2,
        postoptimize=false,
        fit_prior=false,
        keep_all=true,
    )
    r = optimize(
        m, alg, [1.234, 2.345], options=options, callback=sol -> (cb_counter[] += 1)
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
        options=BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=2,
            ctol=1e-12,
            ftol=1e-12,
            initialize=false,
            postoptimize=false,
            fit_prior=false,
        ),
    )
    @test r2.iter == 2
    @test r2.maxiter_reached

    # A long run converges before maxiter.
    r3 = optimize(
        m,
        alg,
        [1.234, 2.345],
        options=BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=200,
            ctol=1e-4,
            ftol=1e-4,
            ninit=2,
            postoptimize=false,
            fit_prior=false,
        ),
    )
    @test !r3.maxiter_reached
    @test r3.convstate.converged

    # verbose mode runs without error.
    rv = optimize(
        m,
        alg,
        [1.234, 2.345],
        options=BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=3,
            ninit=1,
            postoptimize=false,
            fit_prior=false,
            verbose=true,
        ),
    )
    @test rv.iter == 3
end

@testitem "reset!" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra
    using NonconvexCore: tovecmodel, Workspace

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=4,
        ctol=1e-4,
        ftol=1e-12,   # never converge early: deterministic evaluation counts
        ninit=2,
        postoptimize=false,
        fit_prior=false,
    )
    vm, x0, _ = tovecmodel(m, [1.234, 2.345])
    w = Workspace(vm, alg, x0; options=options)
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

@testitem "Capabilities" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra
    using NonconvexCore: AbstractOptimizer, ConstraintCapabilities, solver_capabilities

    f(x::AbstractVector) = sqrt(x[2])

    # A sub-optimizer that declares it does not support semidefinite
    # constraints, used to test capability forwarding through BayesOptAlg.
    struct NoSDOptimizer <: AbstractOptimizer end
    function NonconvexCore.solver_capabilities(::NoSDOptimizer)
        return ConstraintCapabilities(; linear_sd=false, nonlinear_sd=false)
    end

    caps = solver_capabilities(BayesOptAlg(IpoptAlg()))
    @test caps.linear_eq === true
    @test caps.nonlinear_eq === true
    @test caps.linear_ineq === true
    @test caps.nonlinear_ineq === true
    @test caps.box === true
    @test caps.continuous === true
    @test caps.linear_sd === false        # forwarded from IpoptAlg's declaration
    @test caps.nonlinear_sd === false     # forwarded from IpoptAlg's declaration
    @test caps.binary === false           # forwarded from IpoptAlg's declaration
    @test caps.integer === false          # forwarded from IpoptAlg's declaration
    @test caps.infinite_bounds === true   # forwarded from IpoptAlg's declaration

    # sd is forwarded from the sub-solver.
    caps2 = solver_capabilities(BayesOptAlg(NoSDOptimizer()))
    @test caps2.linear_sd === false
    @test caps2.nonlinear_sd === false

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
            options=BayesOptOptions(sub_options=nothing),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("semidefinite", sprint(showerror, err))
end

@testitem "MLE controls" begin
    using NonconvexBayesian, NonconvexIpopt, Test

    f(x::AbstractVector) = sqrt(x[2])

    # The default hyperparameter optimizer is Ipopt.
    @test NonconvexBayesian._fit_algorithm(nothing, 5) isa Tuple

    # fit_max_iter, fit_bounds and fit_optimizer are accepted and refits run.
    s = NonconvexBayesian.surrogate(
        f,
        [1.234, 2.345];
        every=1,
        skip=0,
        fit_max_iter=3,
        fit_bounds=(lb=-10.0, ub=10.0),
        fit_optimizer=(IpoptAlg(), IpoptOptions(max_iter=3, print_level=0)),
    )
    s.mode = :exact
    s([1.5, 2.0])
    @test s.counter == 1  # the factory no longer double-evaluates x0
    # Invalid fit_bounds lengths are rejected; valid vectors are broadcast.
    @test_throws ArgumentError NonconvexBayesian._expand_to([1.0, 2.0], 3)
    @test NonconvexBayesian._expand_to([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]
end

@testitem "Integer capability error" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])

    m = Model()
    set_objective!(m, f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    setinteger!(m, 1, true)
    err = try
        optimize(
            m,
            BayesOptAlg(IpoptAlg()),
            [1.234, 2.345],
            options=BayesOptOptions(sub_options=nothing),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("integer", sprint(showerror, err))
end

@testitem "First-order BO with an expensive constraint" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:expensive])
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=40,
        ctol=1e-4,
        ninit=8,
        postoptimize=true,  # polish the incumbent on the exact model
        fit_prior=false,
        first_order=true,
        grad_source=:forwarddiff,
    )
    r = optimize(m, alg, [1.234, 2.345], options=options)
    # Smoke test for the gradient-enhanced constraint path (the inexact
    # gradients can land the BO loop slightly off, and postoptimize polishes).
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
end

@testitem "Initialization with infinite bounds" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])

    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, fill(-Inf, 2), fill(Inf, 2))
    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=2,
        ninit=2,
        initialize=true,
        postoptimize=false,
        fit_prior=false,
    )
    r = @test_logs (:warn, r"finite") optimize(m, alg, [0.5, 0.5], options=options)
    @test r.iter == 2
end

@testitem "Trust-region machinery" begin
    using NonconvexBayesian, Test

    # Bounds of the hyperrectangle, clamped to the variable bounds.
    @test NonconvexBayesian._tr_bounds([0.5, 0.5], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]) ==
        ([0.0, 0.0], [1.0, 1.0])
    @test NonconvexBayesian._tr_bounds([0.9, 0.5], [0.4, 0.4], [0.0, 0.0], [1.0, 1.0]) ==
        ([0.7, 0.3], [1.0, 0.7])

    # TuRBO success/failure rule.
    L_init = [0.8, 0.8]
    L_min = [0.05, 0.05]
    # A success below τ_success does not grow the region (the counter counts).
    L, restart, s, f = NonconvexBayesian._update_tr(
        [0.1, 0.1], true, 1, 0, 3, 4, L_init, L_min
    )
    @test L == [0.1, 0.1] && !restart && s == 2 && f == 0
    # τ_success consecutive improvements double the region (capped at L_init)
    # and reset the success counter.
    L, restart, s, f = NonconvexBayesian._update_tr(
        [0.1, 0.1], true, 2, 0, 3, 4, L_init, L_min
    )
    @test L == [0.2, 0.2] && !restart && s == 0 && f == 0
    L, restart, s, f = NonconvexBayesian._update_tr(
        [0.4, 0.4], true, 2, 0, 3, 4, L_init, L_min
    )
    @test L == L_init && !restart && s == 0 && f == 0
    # Failures reset the success counter...
    L, restart, s, f = NonconvexBayesian._update_tr(
        [0.4, 0.4], false, 3, 0, 3, 4, L_init, L_min
    )
    @test L == [0.4, 0.4] && !restart && s == 0 && f == 1
    # ... and τ_fail consecutive failures halve the region...
    L, restart, s, f = NonconvexBayesian._update_tr(
        [0.4, 0.4], false, 0, 3, 3, 4, L_init, L_min
    )
    @test L == [0.2, 0.2] && !restart && s == 0 && f == 0
    # ... restarting when it collapses below L_min.
    L, restart, s, f = NonconvexBayesian._update_tr(
        [0.08, 0.08], false, 0, 3, 3, 4, L_init, L_min
    )
    @test L == L_init && restart && s == 0 && f == 0
end

@testitem "Trust-region optimization" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    # The standard problem converges with the trust region enabled.
    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=40,
        ctol=1e-4,
        ninit=5,
        postoptimize=false,
        fit_prior=false,
        trust_region=true,
    )
    r = optimize(m, alg, [1.234, 2.345], options=options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test r.n_restarts >= 0
    # The trust-region boxing must not mutate the user's model bounds.
    @test getmin(m) == [1e-4, 1e-4]
    @test getmax(m) == [10.0, 10.0]

    # A linear objective with a tight trust region forces a deterministic
    # restart once improvements stall at the boundary.
    f2(x) = x[2]
    m2 = Model()
    set_objective!(m2, f2, flags=[:expensive])
    addvar!(m2, [0.0, 0.0], [1.0, 1.0])
    options2 = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=25,
        ctol=1e-4,
        initialize=false,
        postoptimize=false,
        fit_prior=false,
        trust_region=true,
        tr_length_init=0.1,
        tr_min_factor=0.9,
        tr_success_max=3,
        tr_fail_max=1,
    )
    r2 = optimize(m2, alg, [0.5, 0.5], options=options2)
    @test r2.minimum < 1e-3  # x2 driven to the lower bound
    @test r2.n_restarts >= 1

    # Trust regions require finite variable bounds.
    m3 = Model()
    set_objective!(m3, f2, flags=[:expensive])
    addvar!(m3, fill(-Inf, 2), fill(Inf, 2))
    options3 = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=2,
        initialize=false,
        postoptimize=false,
        fit_prior=false,
        trust_region=true,
    )
    err = try
        optimize(m3, alg, [0.5, 0.5], options=options3)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("finite", sprint(showerror, err))

    # Trust-region option validation.
    @test_throws ErrorException BayesOptOptions(
        sub_options=IpoptOptions(), tr_length_init=1.5
    )
    @test_throws ErrorException BayesOptOptions(
        sub_options=IpoptOptions(), tr_min_factor=0.0
    )
    @test_throws ErrorException BayesOptOptions(
        sub_options=IpoptOptions(), tr_success_max=0
    )
    @test_throws ErrorException BayesOptOptions(sub_options=IpoptOptions(), tr_fail_max=0)
end

@testitem "First-order BO with an expensive equality constraint" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27, flags=[:expensive])
    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0),
        maxiter=20,
        ctol=1e-4,
        ninit=5,
        postoptimize=true,
        fit_prior=false,
        first_order=true,
        grad_source=:forwarddiff,
    )
    r = optimize(m, alg, [1.234, 2.345], options=options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-3
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-3
end

@testitem "Reproducibility" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra

    # Two identical runs must produce identical results: the algorithm draws
    # no random numbers (deterministic Sobol sequences only).
    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    function run_once()
        m = Model()
        set_objective!(m, f, flags=[:expensive])
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))
        options = BayesOptOptions(
            sub_options=IpoptOptions(print_level=0),
            maxiter=8,
            ninit=3,
            postoptimize=false,
            fit_prior=false,
        )
        return optimize(m, BayesOptAlg(IpoptAlg()), [1.234, 2.345], options=options)
    end

    r1 = run_once()
    r2 = run_once()
    @test r1.minimum == r2.minimum
    @test r1.minimizer == r2.minimizer
    @test r1.fcalls == r2.fcalls
end
