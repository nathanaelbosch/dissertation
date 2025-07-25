using ProbNumDiffEq, LinearAlgebra, OrdinaryDiffEq#, Plots
using ProbNumDiffEq.DataLikelihoods
using Optimization, OptimizationOptimJL, OptimizationBBO
using Random
using JLD2, SimpleUnPack, SciMLBase
@load "scripts/2024-07-28-fenrirtempering_data.jld2" tspan true_sol sol odedata times optsolu optsol2u appxsol appxsol2 p u0 param_trajectory param_trajectory2

DIR = @__DIR__

using OrdinaryDiffEq, ProbNumDiffEq
using CairoMakie, TuePlots, ColorSchemes
using Dissertation
using Statistics

import Dissertation: plot_sol!, plot_errs!

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
        nrows=4, ncols=4,
    )
))

fig = Figure()

ax1 = Axis(
    fig[1, 1],
    xticks=[tspan...],
    title="Initial trajectory",
    xlabel="t",
    ylabel="y(t)",
)
plot_sol!(ax1, true_sol; markersize=0, colors=(:black, :black), linestyle=:dash, label="true solution")
plot_sol!(ax1, sol; markersize=0, colors=COLORS, label="initial parameters")
scatter!(ax1, times, stack(odedata)[:], color=:gray, markersize=4, label="observations")
ylims!(ax1, -2.5, 2.5)


ax2 = Axis(
    fig[1, 2],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Optimized trajectory",
    xlabel="t",
    ylabel="",
)
p_mle = optsolu[1:3]
u0_mle = optsolu[4:5]
sol2 = appxsol
plot_sol!(ax2, true_sol; markersize=0, colors=(:black, :black), linestyle=:dash, label="true solution")
plot_sol!(ax2, sol2; markersize=0, colors=COLORS, label="optimized trajectory")
scatter!(ax2, times, stack(odedata)[:], color=:gray, markersize=4, label="observations")
ylims!(ax2, -2.5, 2.5)


labels = ["α", "β", "γ", "y₁", "y₂", "log(σ²)"]
gl = fig[1, 3] = GridLayout()
_p = [p..., u0...]
for i in 1:length(param_trajectory[1])
    ax = Axis(
        gl[
            (i-1)÷3+1,
            (i-1)%3,
        ],
        title=labels[i],
        xticklabelsvisible=((i - 1) ÷ 3 == 1),
        xticks=0:1000:length(param_trajectory),
    )
    if i <= length(_p)
        hlines!(ax, [_p[i]], color=:black, linestyle=:dash, linewidth=0.5)
    end
    lines!(ax, 1:length(param_trajectory), [p[i] for p in param_trajectory])
end
rowgap!(gl, 2)
colgap!(gl, 2)

# path = joinpath(DIR, "..", "../figures/fenrirtempering1.pdf")
# save(path, fig)
# @info "Saved figure to $path"
# fig = Figure()

ax1 = Axis(
    fig[2, 1],
    xticks=[tspan...],
    title="Initial trajectory",
    xlabel="t",
    ylabel="y(t)",
)
plot_sol!(ax1, true_sol; markersize=0, colors=(:black, :black), linestyle=:dash, label="true solution")
plot_sol!(ax1, sol; markersize=0, colors=COLORS, label="initial parameters")
scatter!(ax1, times, stack(odedata)[:], color=:gray, markersize=4, label="observations")
ylims!(ax1, -2.5, 2.5)


ax2 = Axis(
    fig[2, 2],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Optimized trajectory",
    xlabel="t",
    ylabel="",
)
p_mle2 = optsol2u[1:3]
u0_mle2 = optsol2u[4:5]
sol2 = appxsol2
plot_sol!(ax2, true_sol; markersize=0, colors=(:black, :black), linestyle=:dash, label="true solution")
plot_sol!(ax2, sol2; markersize=0, colors=COLORS, label="optimized trajectory")
scatter!(ax2, times, stack(odedata)[:], color=:gray, markersize=4, label="observations")
ylims!(ax2, -2.5, 2.5)


labels = ["α", "β", "γ", "y₁", "y₂", "log(σ²)"]
gl = fig[2, 3] = GridLayout()
_p = [p..., u0...]
for i in 1:length(param_trajectory2[1])
    ax = Axis(
        gl[
            (i-1)÷3+1,
            (i-1)%3,
        ],
        title=labels[i],
        xticklabelsvisible=((i - 1) ÷ 3 == 1),
        xticks=0:8000:length(param_trajectory2),
    )
    if i <= length(_p)
        hlines!(ax, [_p[i]], color=:black, linestyle=:dash, linewidth=0.5)
        lines!(ax, 1:length(param_trajectory2), [p[i] for p in param_trajectory2])
    else
        lines!(ax, 1:length(param_trajectory2), [log(p[i]) for p in param_trajectory2])
    end
end
rowgap!(gl, 2)
colgap!(gl, 2)

Label(fig[1, 0], "Optimizing σ", rotation=pi / 2, tellheight=false)
Label(fig[2, 0], "Scheduling σ", rotation=pi / 2, tellheight=false)

# Add legend to the bottom
Legend(
    fig[3, 1:2],
    [LineElement(color=:black, linestyle=:dash, linewidth=1),
     MarkerElement(color=:gray, marker=:o, markersize=4),
     [LineElement(color=COLORS[1],
                  linewidth=1,
                  points=Point2f[(0, 0.66), (1, 0.66)]),
      LineElement(color=COLORS[2],
                  linewidth=1,
                  points=Point2f[(0, 0.33), (1, 0.33)])],
    ],
    ["True trajectory",
     "Observations",
     "Estimated trajectory",
    ],
    tellheight=true,
    orientation=:horizontal,
    patchsize = (10, 10),
    colgap=10,
)
rowgap!(fig.layout, 2)


# path = joinpath(DIR, "..", "../figures/fenrirtempering2.pdf")
path = joinpath(DIR, "..", "../figures/fenrirtempering.pdf")
save(path, fig)
@info "Saved figure to $path"
