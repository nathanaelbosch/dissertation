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
        nrows=2, ncols=4,
    )
))

function vanderpol(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * ((1 - u[1]^2) * u[2] - u[1])
    return nothing
end
u0 = [2.0, 0.0]
tspan = [0.0, 3.6]
p = [1e3]
prob = ODEProblem(vanderpol, u0, tspan, p)

# ref = solve(remake(prob, u0=big.(u0)), Vern9(), abstol=1e-16, reltol=1e-16)
sol = solve(prob, EK1(), abstol=1e-6, reltol=1e-6,
    # controller=PIController(7 // (10*3), 2 // (5*3))
    controller = IController(),
)
# sol = solve(prob, EK1(), adaptive=false, dt=9e-6)
@info "" sol.destats


ylims = -5, 5

fig = Figure()
ax1 = Axis(
    fig[1, 1],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Van-der-Pol ODE",
    xlabel="t",
    ylabel="y(t)",
)
# plot_sol!(ax1, ref; markersize=0, colors=(:black,:black), linestyle=:dash)
plot_sol!(ax1, sol; markersize=0, colors=COLORS)
xlims!(tspan...)
ylims!(ylims...)

ax2 = Axis(
    fig[1, 2],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Error and error estimate",
    xlabel="t",
    ylabel="|y(t)-y*(t)|",
    yscale=log10,
)
plot_errs!(ax2, sol, ref; colors=COLORS, markersize=0, logscale=true)
xlims!(tspan...)
# ylims!(-1e-3, 1e-3)
ylims!(1e-10, 1e0)


ax3 = Axis(
    fig[1, 3],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Step size",
    xlabel="t",
    ylabel="Δt",
    yscale=log10,
)
step_sizes = sol.t[2:end] .- sol.t[1:end-1]
lines!(
    ax3,
    sol.t[1:end-1],
    step_sizes;
    color=COLORS[3],
    # yscale=log10,
)
xlims!(tspan...)


resize_to_layout!(fig)
path = joinpath(DIR, "..", "../figures/stepsizeadaptation.pdf")
save(path, fig)
@info "Saved figure to $path"
