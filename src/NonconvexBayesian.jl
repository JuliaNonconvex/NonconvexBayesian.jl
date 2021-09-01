module NonconvexBayesian

export BayesOptOptions, BayesOptAlg

using Reexport, AbstractGPs, IntervalArithmetic, Parameters
using Sobol, ChainRulesCore
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult, AbstractOptimizer, getinit
using NonconvexCore: VectorOfFunctions, IneqConstraint, EqConstraint
using NonconvexCore: Objective, flatten, @params
import NonconvexCore: optimize!, Workspace
using NonconvexIpopt

include("bayesian.jl")

end
