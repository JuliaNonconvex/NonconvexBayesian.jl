@testitem "Exported API: types, workspace and result fields" begin
    using NonconvexBayesian, NonconvexIpopt, NonconvexCore, Test, LinearAlgebra
    using NonconvexCore: tovecmodel
    using NonconvexCore: tovecmodel

    # acquisition types share the AbstractAcquisition interface
    @test LCB() isa AbstractAcquisition
    @test LCB(2.0) isa AbstractAcquisition
    @test GPUCB(1.0) isa AbstractAcquisition
    @test GPUCB(1.0; schedule=t -> sqrt(t + 1)) isa AbstractAcquisition
    @test EI() isa AbstractAcquisition
    @test EI(0.05) isa AbstractAcquisition
    @test PI() isa AbstractAcquisition
    @test PI(0.05) isa AbstractAcquisition

    f(x) = sqrt(x[2])
    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> (2x[1] + 0)^3 - x[2])

    # surrogate constructors produce AbstractGPSurrogates of the right kind
    x0 = [1.0, 2.0]
    s0 = surrogate(f, x0)
    s1 = first_order_surrogate(f, x0)
    s2 = co_kriging_surrogate(f, x0)
    @test s0 isa ZeroOrderGPSurrogate
    @test s1 isa FirstOrderGPSurrogate
    @test s2 isa CoKrigingSurrogate
    @test all(s -> s isa AbstractGPSurrogate, (s0, s1, s2))

    # optimize returns a BayesOptResult
    alg = BayesOptAlg(IpoptAlg())
    options = BayesOptOptions(
        sub_options=IpoptOptions(print_level=0), maxiter=8, ninit=3, verbose=false
    )
    r = optimize(m, alg, x0, options=options)
    @test r isa BayesOptResult
    @test length(r.minimizer) == 2
    @test isfinite(r.minimum)
    @test r.fcalls >= 0
    @test r.iter >= 1
    @test r.surrogates !== nothing

    # the workspace protocol: build + reset! (the Workspace constructor
    # takes a VecModel, so convert the model first)
    vm, x0v, _ = tovecmodel(m, x0)
    w = Workspace(vm, alg, x0v; options=options)
    @test w isa BayesOptWorkspace
    @test w.options === options
    @test length(w.surrogates) == 1   # only the expensive objective
    NonconvexCore.reset!(w)
    @test w isa BayesOptWorkspace
end

@testitem "surrogate_model and update_surrogates! round trip" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra
    using NonconvexCore: tovecmodel, getinit

    f(x) = sqrt(x[2])
    g(x) = (2x[1] + 0)^3 - x[2]
    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, g, flags=[:expensive])
    vm, x0, _ = tovecmodel(m, [1.234, 2.345])

    sm, surrogates = surrogate_model(vm)
    @test length(surrogates) == 2                      # objective + constraint
    @test all(s -> s isa ZeroOrderGPSurrogate, surrogates)
    @test :expensive in sm.objective.flags
    @test length(sm.eq_constraints.fs) == 0
    @test length(sm.ineq_constraints.fs) == 1
    @test sm.box_min == vm.box_min && sm.box_max == vm.box_max

    # acquisition selection propagates to the objective surrogate
    _, surr_ei = surrogate_model(vm; acquisition=EI())
    @test surr_ei[1].acquisition isa EI

    # update_surrogates! records the true observations and reports the
    # true model values at the candidate
    x = [1.5, 2.0]
    o, i, e = update_surrogates!(sm, surrogates, x)
    @test o ≈ f(x)
    @test i ≈ [g(x)]
    @test isempty(e)
    @test all(s -> s.mode === :interval, surrogates)
    @test length(surrogates[1].X) >= 2                 # initial + observation
end

@testitem "Constraint transformation variants: first-order and co-kriging" begin
    using NonconvexBayesian, NonconvexIpopt, Test, LinearAlgebra
    using NonconvexCore: tovecmodel

    f(x) = sqrt(x[2])
    h(x) = [x[1] + x[2] - 1]                    # dim-1 expensive equality
    m = Model()
    set_objective!(m, f, flags=[:expensive])
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_eq_constraint!(m, h, flags=[:expensive])
    vm, _, _ = tovecmodel(m, [1.234, 2.345])

    # first-order: FirstOrderGPSurrogates everywhere, same transformation.
    # Each expensive equality becomes ONE 2-block inequality of twice the
    # dimension (the [lo; -hi] block is a single constraint).
    sm1, surr1 = surrogate_model(vm; first_order=true)
    @test all(s -> s isa FirstOrderGPSurrogate, surr1)
    @test length(sm1.eq_constraints.fs) == 0
    @test length(sm1.ineq_constraints.fs) == 1
    @test sm1.ineq_constraints.fs[1].dim == 2
    @test all(iszero, sm1.ineq_constraints.fs[1].rhs)

    # co-kriging: CoKrigingSurrogates everywhere
    sm2, surr2 = surrogate_model(vm; co_kriging=true)
    @test all(s -> s isa CoKrigingSurrogate, surr2)
    @test length(sm2.ineq_constraints.fs) == 1
    @test sm2.ineq_constraints.fs[1].dim == 2

    # std_multiple passes through to the LCB acquisition of the surrogates
    _, surr3 = surrogate_model(vm; std_multiple=2.0)
    @test all(s -> s.acquisition.η == 2.0, surr3)
end
