DIR = @__DIR__

using ProbNumDiffEq
using CairoMakie, TuePlots, ColorSchemes, LaTeXStrings
using Dissertation


function lv(u, p, t)
    x, y = u
    a, b, c, d = p
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
    return [dx, dy]
end
u0 = [1.0, 1.0]
d = 2
tspan = [0.0, 30.0]
p = [2 / 3, 4 / 3, 1, 1]
prob = ODEProblem(lv, u0, tspan, p)
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
ylims = 0, 2
COLORS = ColorSchemes.tableau_10.colors[[1, 2, 5]]
function plot_sol!(ax, sol; markersize=3, strokewidth=0.1, kwargs...)
    plotrange = range(sol.t[begin], sol.t[end], length=200)
    _sol = sol(plotrange)
    if !(sol isa ProbNumDiffEq.ProbODESolution)
        for i in 1:d
            scatterlines!(
                ax, _sol.t, [u[i] for u in _sol.u];
                color=COLORS[i],
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
    else
        for i in 1:d
            lines!(
                ax, _sol.t, [mean(u)[i] for u in _sol.u];
                color=COLORS[i],
                label=L"y_$i",
                kwargs...)
            scatter!(
                ax, sol.t, [u[i] for u in sol.u];
                color=COLORS[i],
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
    end
end
fig = Figure()
ax1 = Axis(
    fig[1, 1],
    yticks=[0, 2],
    xticks=tspan,
    title="Lotka-Volterra ODE",
    xlabel="t",
    ylabel="y(t)",
)
plot_sol!(ax1, ref; markersize=0)
ylims!(ax1, ylims...)
xlims!(ax1, tspan...)

ax2 = Axis(
    fig[1, 2],
    yticks=[0, 2],
    xticks=tspan,
    title="Probabilistic ODE solution",
    xlabel="t",
    ylabel="",
)
sol = solve(prob, EK1(
    order=3,
    diffusionmodel=FixedDiffusion(),
    initialization=SimpleInit(),
); adaptive=false, dt=0.5)
plot_sol!(ax2, sol; markersize=2)
ylims!(ax2, ylims...)
xlims!(ax2, tspan...)

ax3 = Axis(
    fig[1, 3],
    # yticks=[0, 2],
    xticks=tspan,
    title="Error and error estimate",
    xlabel="t",
    ylabel="y(t)  - y*(t)",
)
function plot_errs!(ax, sol, ref; markersize=3, strokewidth=0.1, kwargs...)
    plotrange = range(sol.t[begin], sol.t[end], length=200)
    _sol = sol(plotrange)
    _ref = ref(plotrange)
    ref = ref(sol.t)
    for i in 1:d
        band!(
            ax, _sol.t,
            [-1.96std(u)[i] for u in _sol.u],
            [1.96std(u)[i] for u in _sol.u];
            color=(COLORS[i], 0.4),
            markersize,
            strokewidth,
            label=L"y_$i",
            kwargs...)
    end
    for i in 1:d
        lines!(
            ax, _sol.t, [mean(u)[i] - r[i] for (u, r) in zip(_sol.u, _ref.u)];
            color=COLORS[i],
            label=L"y_$i",
            kwargs...)
        scatter!(
            ax, sol.t, [u[i] - r[i] for (u, r) in zip(sol.u, ref)];
            color=COLORS[i],
            markersize,
            strokewidth,
            label=L"y_$i",
            kwargs...)
    end
end
plot_errs!(ax3, sol, ref; markersize=2)
# ylims!(ax, ylims...)
xlims!(ax3, tspan...)


path = joinpath(DIR, "..", "../figures/odefilterdemo.pdf")
save(path, fig)
@info "Saved figure to $path"
