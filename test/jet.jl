using JET, NonconvexBayesian

# Run JET analysis on the NonconvexBayesian package.
# Start with a permissive config to catch obvious issues without
# blocking CI on known type-instabilities or missing methods.
JET.test_package(NonconvexBayesian; target_modules=(NonconvexBayesian,))
