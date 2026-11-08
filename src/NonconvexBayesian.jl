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
    surrogate,
    first_order_surrogate,
    surrogate_model,
    update_surrogates!,
    get_lo

using Reexport, AbstractGPs, IntervalArithmetic, Parameters
using Sobol, ChainRulesCore
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult, AbstractOptimizer, getinit
using NonconvexCore: VectorOfFunctions, IneqConstraint, EqConstraint
using NonconvexCore: Objective, flatten, @params
using NonconvexCore: NoCallback, Solution, ConvergenceState
import NonconvexCore: optimize!, Workspace, reset!

include("bayesian.jl")

end
