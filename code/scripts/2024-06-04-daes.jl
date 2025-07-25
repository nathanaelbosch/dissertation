DIR = @__DIR__

using ProbNumDiffEq, OrdinaryDiffEq
using CairoMakie, TuePlots, ColorSchemes, LaTeXStrings
using Dissertation
using LinearAlgebra


function rober_mm(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    return nothing
end
M = [
    1.0 0 0
    0 1.0 0
    0 0 0
]
f_mm = ODEFunction(rober_mm, mass_matrix=M)
u0 = [1.0, 0.0, 0.0]
tspan = [0.0, 1e5]
p = (0.04, 3e7, 1e4)
prob = ODEProblem(f_mm, u0, tspan, p)
ref = solve(remake(prob, u0=big.(u0)), RadauIIA5(), abstol=1e-30, reltol=1e-30)
d = length(u0)


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
# ylims = 0, 2
COLORS = ColorSchemes.tableau_10.colors[[1, 2, 5]]
function plot_sol!(axes, sol; markersize=3, strokewidth=0.1, kwargs...)
    @info "plot_sol!"
    plotrange = range(sol.t[begin], sol.t[end], length=200)
    _sol = sol(plotrange)
    if !(sol isa ProbNumDiffEq.ProbODESolution)
        for i in 1:d
            scatterlines!(
                axes[i], _sol.t, [u[i] for u in _sol.u];
                color=COLORS[i],
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
    else
        for i in 1:d
            lines!(
                axes[i], _sol.t, [mean(u)[i] for u in _sol.u];
                color=COLORS[i],
                label=L"y_$i",
                kwargs...)
            scatter!(
                axes[i], sol.t, [u[i] for u in sol.u];
                color=COLORS[i],
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
    end
end
fig = Figure()
g1 = fig[1, 1] = GridLayout()
y2max = 3e-6
axes1 = [
    Axis(g1[i, 1],
         yticks=i == 2 ? ([0, y2max], ["0", "3×10⁻⁶"]) : [0, 1],
        xticks=i == d ? (tspan, ["0", "10⁵"]) : (tspan, ["", ""]),
         xlabel=i == d ? "t" : "",
         # ylabel=["y₁(t)", "y₂(t)", "y₃(t)"][i],
         ylabel=[L"y_1(t)", L"y_2(t)", L"y_3(t)"][i],
         title=i == 1 ? "Robertson DAE" : "",
    )
    for i in 1:d
]
ylims!(axes1[2], 0, y2max)
plot_sol!(axes1, ref; markersize=0)
# xlims!(ax1, tspan...)
# ylims!(ax1, 1e-10, 1)

g2 = fig[1, 2] = GridLayout()
axes2 = [
    Axis(g2[i, 1],
        yticks=i == 2 ? ([0, y2max], ["", ""]) : ([0, 1], ["", ""]),
        xticks=i == d ? (tspan, ["0", "10⁵"]) : (tspan, ["", ""]),
        xlabel=i == d ? "t" : "",
        # ylabel=["y₁(t)", "y₂(t)", "y₃(t)"][i],
        ylabel="",
        title=i == 1 ? "Probabilistic DAE solution" : "",
    )
    for i in 1:d
]
ylims!(axes2[2], 0, y2max)
sol = solve(prob, EK1(
    order=1,
    # diffusionmodel=FixedDiffusion(),
    initialization=SimpleInit(),
))
plot_sol!(axes2, sol; markersize=0)

function _plot_errs!(axes, sol, ref; markersize=3, strokewidth=0.1, kwargs...)
    ref = ref(sol.t)
    for i in 1:d
        band!(
            axes[i], sol.t,
            [1e-20 for u in sol.u],
            [1.96std(u)[i] for u in sol.pu];
            color=(COLORS[i], 0.3),
            markersize,
            strokewidth,
            label=L"y_$i",
            kwargs...)
    end
    for i in 1:d
        lines!(
            axes[i], sol.t, [abs.(u[i] - r[i]) for (u, r) in zip(sol.u, ref)];
            color=COLORS[i],
            label=L"y_$i",
            kwargs...)
    end
    for i in 1:d
        lines!(
            axes[i], sol.t, [1.96std(u)[i] for (u, r) in zip(sol.pu, ref)];
            color=(COLORS[i], 0.5),
            label=L"y_$i",
            linestyle=:dash,
            width=2,
            kwargs...)
    end
end
function plot_errs!(axes, sol, ref; markersize=3, strokewidth=0.1, kwargs...)
    ref = ref(sol.t)
    for i in 1:d
        band!(
            axes[i], sol.t,
            [-1.96std(u)[i] for u in sol.pu],
            [1.96std(u)[i] for u in sol.pu];
            color=(COLORS[i], 0.4),
            markersize,
            strokewidth,
            label=L"y_$i",
            kwargs...)
    end
    for i in 1:d
        lines!(
            axes[i], sol.t, [u[i] - r[i] for (u, r) in zip(sol.u, ref)];
            color=COLORS[i],
            label=L"y_$i",
            kwargs...)
    end
end
g3 = fig[1, 3] = GridLayout()
axes3 = [Axis(
    g3[i, 1],
    yticks=([1e-6, 1e-12, 1e-18], ["10⁻⁶", "10⁻¹²", "10⁻¹⁸"]),
    # yticks=([1e-6, 1e-18], ["10⁻⁶", "10⁻¹⁸"]),
    # yticks=i == 2 ?
    #     ([-1e-9, 0, 1e-9], ["-10⁻⁹", "0", "10⁻⁹"]) :
    #     ([-1e-7, 0, 1e-7], ["-10⁻⁷", "0", "10⁻⁷"]),
    xticks=i == d ? (tspan, ["0", "10⁵"]) : (tspan, ["", ""]),
    title=i == 1 ? "Error and error estimate" : "",
    xlabel=i == d ? "t" : "",
    ylabel="",
    yscale=log10
) for i in 1:d]
[ylims!(ax, 1e-18, 1e-6) for ax in axes3]
# ylims!(axes3[1], -1e-7, 1e-7)
# ylims!(axes3[2], -1e-9, 1e-9)
# ylims!(axes3[3], -1e-7, 1e-7)
_plot_errs!(axes3, sol, ref; markersize=2)
# xlims!(ax3, tspan...)

rowgap!(g1, 5)
rowgap!(g2, 5)
rowgap!(g3, 5)

path = joinpath(DIR, "..", "../figures/daes.pdf")
save(path, fig)
@info "Saved figure to $path"
