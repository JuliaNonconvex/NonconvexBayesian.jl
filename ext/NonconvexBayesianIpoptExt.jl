module NonconvexBayesianIpoptExt

using NonconvexBayesian, NonconvexIpopt

# Default hyperparameter optimizer: Ipopt with a small iteration budget.
NonconvexBayesian._fit_algorithm(
    ::NonconvexBayesian.NoFitOptimizer,
    max_iter,
) = (IpoptAlg(), IpoptOptions(max_iter = max_iter, print_level = 0))

end
