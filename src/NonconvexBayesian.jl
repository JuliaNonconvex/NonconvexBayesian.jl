module NonconvexBayesian

export BayesOptOptions,
    BayesOptAlg,
    BayesOptWorkspace,
    BayesOptResult,
    AbstractAcquisition,
    LCB,
    GPUCB,
    EI,
    PI,
    AbstractGPSurrogate,
    ZeroOrderGPSurrogate,
    FirstOrderGPSurrogate,
    CoKrigingSurrogate,
    surrogate,
    first_order_surrogate,
    co_kriging_surrogate,
    surrogate_model,
    update_surrogates!,
    get_lo

using Reexport, AbstractGPs, IntervalArithmetic, Parameters
using NonconvexIpopt: IpoptAlg, IpoptOptions
using Sobol, ChainRulesCore
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult, AbstractOptimizer, getinit
using NonconvexCore: VectorOfFunctions, IneqConstraint, EqConstraint
using NonconvexCore: Objective, flatten, @params
using NonconvexCore: NoCallback, Solution, ConvergenceState
import NonconvexCore: optimize!, Workspace, reset!

include("bayesian.jl")
include("cokriging.jl")

end
