DIR = @__DIR__

using OrdinaryDiffEq
using CairoMakie, TuePlots, ColorSchemes
using Dissertation


function logistic(u, p, t)
    p[1] * u^2 * (1 - u)
end
u0 = 0.1
tspan = [0.0, 30.0]
p = [1]

function logistic(u, p, t)
    -50(u - cos(t))
end
u0 = 0.0
tspan = [0, π/4]
tspanl = ["0", "π/4"]

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
lax = Axis(
    fig[1, 1], yticks=[0, 1], xticks=(tspan, tspanl),
    title="Stiff ODE: ẏ = 50(y-cos(t))",
    xlabel="t",
    ylabel="y(t)",
)
plot_vf!(lax)
plot_sol!(lax, ref; markersize=0)
ylims!(lax, ylims...)
xlims!(lax, tspan...)


function mkplot!(ax, alg; adaptive=false, kwargs...)
    sol = solve(prob, alg(); adaptive=adaptive, kwargs...)
    plot_sol!(ax, ref; markersize=0, linestyle=:dash, color=:black, label="ref")
    plot_sol!(ax, sol; label="dt = $dt", color=:gray)
    ylims!(ax, ylims...)
    xlims!(ax, tspan...)
    hideydecorations!(ax, ticks=false, label=false)
end

g1 = fig[1, 2] = GridLayout()
alg, dt = Euler, 0.035
mkplot!(
    Axis(
        g1[1,1],
        title="Explicit Euler, Δₜ = $dt",
        xticklabelsvisible=false,
        yticks=[0, 1],
        xticks=(tspan, tspanl),
    ),
    alg; dt=dt)

alg, dt = RK4, 0.05
mkplot!(
    Axis(
        g1[2,1],
        title="Runge–Kutta 4, Δₜ = $dt",
        # xticklabelsvisible=false,
        yticks=[0, 1],
        xticks=(tspan, tspanl),
        xlabel="t",
    ),
    alg; dt=dt)

g2 = fig[1, 3] = GridLayout()

mkplot!(
    Axis(
        g2[1,1],
        title="Runge–Kutta 4, adaptive",
        xticklabelsvisible=false,
        yticks=[0, 1],
        xticks=(tspan, tspanl),
    ),
    RK4; adaptive=true)

dt = 0.1
mkplot!(
    Axis(
        g2[2,1],
        title="RadauIIA5 (implicit), Δₜ = $dt",
        # xticklabelsvisible=false,
        yticks=[0, 1],
        xticks=(tspan, tspanl),
        xlabel="t",
    ),
    RadauIIA5; dt=dt)


rowgap!(g1, 2)
rowgap!(g2, 2)

path = joinpath(DIR, "..", "../figures/rkstiff.pdf")
save(path, fig)
@info "Saved figure to $path"
