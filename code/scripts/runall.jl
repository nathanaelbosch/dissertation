safeinclude(s) = begin
    @info "Running $s"
    include(s)
end

safeinclude("2024-02-05_prior_figure.jl")
safeinclude("2024-02-20-GPs.jl")
safeinclude("2024-04-17-LTISDE_GPs.jl")
safeinclude("2024-04-19-GMP_regression_example.jl")
safeinclude("2024-04-24-rkexample.jl")
safeinclude("2024-04-29-rkstiff.jl")
safeinclude("2024-05-02-odefilterdemo.jl")
