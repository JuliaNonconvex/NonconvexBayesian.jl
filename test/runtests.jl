using TestItemRunner
using LinearAlgebra, Random

# Deterministic test runs. Single-threaded BLAS removes the main source of
# cross-machine floating-point variation (inside Ipopt's linear solver and
# the GP Cholesky factorizations); the global RNG is seeded as a guard even
# though the package and the tests draw no random numbers (deterministic
# Sobol sequences only).
BLAS.set_num_threads(1)
Random.seed!(0)

@run_package_tests
