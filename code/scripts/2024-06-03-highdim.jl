DIR = @__DIR__

using OrdinaryDiffEq, ProbNumDiffEq
using CairoMakie, TuePlots, ColorSchemes, LaTeXStrings
using Dissertation
using Statistics
using BenchmarkTools

COLORS = ColorSchemes.tableau_10.colors[[1, 2, 5]]

set_theme!(merge(
    Theme(
        palette=(
            # color=Makie.wong_colors(),
            color=ColorSchemes.grays1,
        ),
    ),
    DissertationTheme,
    Theme(
        TuePlots.SETTINGS[:NEURIPS];
        font=false,
        fontsize=true,
        figsize=true,
        thinned=true,
        nrows=1, ncols=2,
    )
))


function get_lorenz(d; diagjac=false)
    function lorenz96(du, u, p, t)
        f = p[1]
        du[1] = (u[2] - u[end-1]) * u[end] - u[1] + f
        du[2] = (u[3] - u[end]) * u[1] - u[2] + f
        du[end] = (u[1] - u[end-2]) * u[end-1] - u[end] + f
        @simd ivdep for i in 3:(length(u)-1)
            du[i] = (u[i+1] - u[i-2]) * u[i-1] - u[i] + f
        end
    end
    F = 8.0
    p = [F]
    u0 = [F for i in 1:d]
    u0[1] += 0.01
    tspan = (0.0, 30.0)

    f = if diagjac
        jac(J, u, p, t) = @simd ivdep for i in 1:d
            J[i, i] = -1
        end
        ODEFunction(lorenz96, jac=jac)
    else
        lorenz96
    end

    prob = ODEProblem(f, u0, tspan, p)
    return prob
end

# prob = get_lorenz(4)
# @mtkbuild sys = modelingtoolkitize(prob)
# prob_jac = ODEProblem(sys, [], (0.0, 1e5), jac=true)
# @variables x[1:4]
# _jac(J, u, p, t) = (J .= Diagonal(prob_jac.f.jac(collect(x), prob_jac.p, 0.0)))
# Symbolics.build_function(
# )

timeit(Alg, ds) = begin
    @info "Timing $Alg"
    [begin
         @info "d=2^$(Int(log2(d)))"
         _prob = get_lorenz(d; diagjac=Alg isa DiagonalEK1)
         @belapsed solve($_prob, $Alg, adaptive=false, dt=1e-2, dense=false, save_everystep=false)
     end
     for d in ds]
end

# ek0_ds = 2 .^ (2:14)
# # ek0_ds = 2 .^ (2:8)
# ek0_times = timeit(EK0(smooth=false), ek0_ds)

# dek1_ds = 2 .^ (2:11)
# # dek1_ds = 2 .^ (2:4)
# dek1_times = timeit(DiagonalEK1(smooth=false), dek1_ds)

# ek1_ds = 2 .^ (2:6)
# # ek1_ds = 2 .^ (2:4)
# ek1_times = timeit(EK1(smooth=false), ek1_ds)


fig = Figure()

ax0 = Axis(
    fig[1, 1];
    ylabel="Time",
    xlabel="State",
    title="Lorenz96 system",
)
# plot the ODE solution
D = 1000
sol = solve(get_lorenz(D), Tsit5(), abstol=1e-9, reltol=1e-9)
M = Array(sol(0:0.1:30))
V = maximum(abs, M)
heatmap!(
    ax0,
    1:D,
    0:0.1:30,
    M,
    colorrange=(-V, V),
    colormap=:bwr,
)
Colorbar(fig[1, 2], limits = (-V, V), colormap = :bwr, size=8)


_ds = [1, 10^6]

ax1 = Axis(
    fig[1, 3];
    yscale=log10, xscale=log10,
    title="Solver comparison",
    ylabel="Runtime [s]",
    xlabel="State dimension",
)
scatterlines!(
    ax1, ek0_ds, ek0_times;
    label="EK0",
    color=COLORS[1],
)
lines!(
    ax1, _ds, 5e-4 * _ds;
    label=L"\mathcal{O}(d)",
    color=(COLORS[1], 0.3),
    linewidth=10,
)

scatterlines!(
    ax1, dek1_ds, dek1_times;
    label="DiagonalEK1",
    color=COLORS[2],
)
lines!(
    ax1, _ds, 4e-3 * _ds;
    label=L"\mathcal{O}(d)",
    color=(COLORS[2], 0.3),
    linewidth=10,
)

scatterlines!(
    ax1, ek1_ds, ek1_times;
    label="EK1",
    color=COLORS[3],
)
lines!(
    ax1, _ds, 1e-4 * _ds .^ 3;
    label=L"\mathcal{O}(d^3)",
    color=(COLORS[3], 0.3),
    linewidth=10,
)
ylims!(ax1, 1e-2, 2e1)
xlims!(ax1, 3e0, 3e4)

# Add a legend
fig[1,4] = Legend(
    fig, ax1,
)

path = joinpath(DIR, "..", "../figures/highdimtimes.pdf")
save(path, fig)
@info "Saved figure to $path"
