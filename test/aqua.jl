@testitem "Aqua" begin
    using Aqua, NonconvexBayesian
    Aqua.test_all(NonconvexBayesian)
end
