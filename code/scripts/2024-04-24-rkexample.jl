DIR = @__DIR__

using OrdinaryDiffEq
using CairoMakie, TuePlots, ColorSchemes
using Dissertation


function logistic(u, p, t)
    p[1] * u * (1 - u)
end
u0 = 0.1
tspan = [0.0, 10.0]
p = [1]
prob = ODEProblem(logistic, u0, tspan, p)
ref = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)


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
ylims = -0.1, 1.1

function plot_sol!(ax, sol; markersize=3, strokewidth=0.1, kwargs...)
    scatterlines!(
        ax, sol.t, sol.u;
        markersize,
        strokewidth,
        kwargs...)
end
function plot_vf!(ax; kwargs...)
    alpha = get(kwargs, :alpha, 1)
    streamplot!(
        ax,
        (P) -> Point2f(
            one(P[1]),
            logistic(P[2], p, P[1]),
        ),
        tspan[1] .. tspan[2],
        ylims[1] .. ylims[2];
        linewidth=0.5,
        arrow_size=2.5,
        density=2.0,
        # colormap=:grays,
        colormap=ColorScheme(range(
            ColorSchemes.RGBA(colorant"gray70", alpha),
            ColorSchemes.RGBA(colorant"gray40", alpha),
            length=5
        )),
        kwargs...
    )
end

fig = Figure()
lax = Axis(fig[1, 1], yticks=[0, 1], xticks=tspan,
    title="Logistic ODE",
    xlabel="t",
    ylabel="y(t)",
)
plot_vf!(lax)
plot_sol!(lax, ref; linestyle=:dash, markersize=0)
ylims!(lax, ylims...)
xlims!(lax, tspan...)

g1 = fig[1, 2] = GridLayout()
DTS = [2.5, 2.0, 1.0, 0.5]
for (i, dt) in enumerate(DTS)
    ax = Axis(g1[i, 1],
              title=i == 1 ? "Explicit Euler" : "",
              xticklabelsvisible=i == length(DTS),
              yticks=[0, 1],
              xticks=tspan,
              xlabel=i == length(DTS) ? "t" : "",
              ylabel="Δₜ = $dt",
              ylabelrotation=4(pi/2),
              ylabelpadding=4,
    )
    sol = solve(prob, Euler(), dt=dt, adaptive=false)
    # plot_vf!(ax; alpha=0.4)
    plot_sol!(ax, ref; markersize=0, linestyle=:dash, color=:black, label="ref")
    plot_sol!(ax, sol; label="dt = $dt", color=:gray)
    ylims!(ax, ylims...)
    xlims!(ax, tspan...)
    # hideydecorations!(ax, ticks=false, label=false)
end

g2 = fig[1, 3] = GridLayout()
for (i, dt) in enumerate(DTS)
    ax = Axis(g2[i, 1],
        title=i == 1 ? "Runge-Kutta 4" : "",
        xticklabelsvisible=i == length(DTS),
        yticks=[0, 1],
        xticks=tspan,
        xlabel=i == length(DTS) ? "t" : "",
    )
    sol = solve(prob, RK4(), dt=dt, adaptive=false)
    # plot_vf!(ax; alpha=0.4)
    plot_sol!(ax, ref; markersize=0, linestyle=:dash, color=:black, label="ref")
    plot_sol!(ax, sol; label="dt = $dt", color=:gray)
    ylims!(ax, ylims...)
    xlims!(ax, tspan...)
    hideydecorations!(ax, ticks=false, label=true)
end

rowgap!(g1, 2)
rowgap!(g2, 2)

path = joinpath(DIR, "..", "../figures/rk.pdf")
save(path, fig)
@info "Saved figure to $path"
