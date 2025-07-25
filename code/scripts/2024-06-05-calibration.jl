DIR = @__DIR__

using OrdinaryDiffEq, ProbNumDiffEq
using CairoMakie, TuePlots, ColorSchemes
using Dissertation
using Statistics
using DiffEqDevTools

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

function lotkavolterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = δ * x * y - γ * y
    return nothing
end
u0 = [1.0, 1.0]
tspan = [0.0, 20.0]
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotkavolterra, u0, tspan, p)

ref = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

dist = Chisq(length(u0))
low, high, mid = quantile(dist, [0.01, 0.99])..., mean(dist)

fig = Figure()
ax1 = Axis(
    fig[1, 0],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Lotka-Volerra ODE",
    xlabel="t",
    ylabel="y(t)",
)
plot_sol!(ax1, ref; markersize=0, colors=COLORS)
xlims!(tspan...)

Label(fig[1, 1], text="y(t)-y*(t)", rotation=pi / 2, tellheight=false)

dt = 1e-2
order = 2
ylims = -5e-5, 5e-5
ylabels = ["-5⋅10⁻⁵", "5⋅10⁻⁵"]
# order = 3
# ylims = -3e-6, 3e-6
# ylabels = ["-3⋅10⁻⁶", "3⋅10⁻⁶"]
# order = 1
# ylims = -3e-3, 3e-3

g1 = fig[1, 2] = GridLayout()
sol = solve(prob, EK0(order=order, diffusionmodel=FixedDiffusion()), adaptive=false, dt=dt)
chi2 = appxtrue(sol, ref).errors[:chi2_final]
@info "EK0 FixedDiffusion" chi2 pdf(dist, chi2)
ax = Axis(
    g1[1, 1],
    xticks=[tspan...],
    yticks=([ylims...], ylabels),
    title="EK0 (global&scalar)",
)
plot_errs!(ax, sol, ref; markersize=0, colors=COLORS)
xlims!(ax, tspan...)
ylims!(ax, ylims...)

sol = solve(prob, EK0(order=order, diffusionmodel=DynamicDiffusion()), adaptive=false, dt=dt)
chi2 = appxtrue(sol, ref).errors[:chi2_final]
@info "EK0 DynamicDiffusion" chi2 pdf(dist, chi2)
ax = Axis(
    g1[2, 1],
    xticks=[tspan...],
    yticks=([ylims...], ylabels),
    title="EK0 (local&scalar)",
)
plot_errs!(ax, sol, ref; markersize=0, colors=COLORS)
xlims!(ax, tspan...)
ylims!(ax, ylims...)

g2 = fig[1, 3] = GridLayout()
sol = solve(prob, EK0(order=order, diffusionmodel=FixedMVDiffusion()), adaptive=false, dt=dt)
chi2 = appxtrue(sol, ref).errors[:chi2_final]
@info "EK0 FixedMVDiffusion" chi2 pdf(dist, chi2)
ax = Axis(
    g2[1, 1],
    xticks=[tspan...], yticks=([ylims...], ["", ""]),
    title="EK0 (global&diagonal)",
)
plot_errs!(ax, sol, ref; markersize=0, colors=COLORS)
xlims!(ax, tspan...)
ylims!(ax, ylims...)

sol = solve(prob, EK0(order=order, diffusionmodel=DynamicMVDiffusion()), adaptive=false, dt=dt)
chi2 = appxtrue(sol, ref).errors[:chi2_final]
@info "EK0 DynamicMVDiffusion" chi2 pdf(dist, chi2)
ax = Axis(
    g2[2, 1],
    xticks=[tspan...], yticks=([ylims...], ["", ""]),
    title="EK0 (local&diagonal)",
)
plot_errs!(ax, sol, ref; markersize=0, colors=COLORS)
xlims!(ax, tspan...)
ylims!(ax, ylims...)

g3 = fig[1, 4] = GridLayout()
sol = solve(prob, EK1(order=order, diffusionmodel=FixedDiffusion()), adaptive=false, dt=dt)
chi2 = appxtrue(sol, ref).errors[:chi2_final]
@info "EK1 FixedDiffusion" chi2 pdf(dist, chi2)
ax = Axis(
    g3[1, 1],
    xticks=[tspan...], yticks=([ylims...], ["", ""]),
    title="EK1 (global&scalar)",
)
plot_errs!(ax, sol, ref; markersize=0, colors=COLORS)
xlims!(ax, tspan...)
ylims!(ax, ylims...)

sol = solve(prob, EK1(order=order, diffusionmodel=DynamicDiffusion()), adaptive=false, dt=dt)
chi2 = appxtrue(sol, ref).errors[:chi2_final]
@info "EK1 DynamicDiffusion" chi2 pdf(dist, chi2)
ax = Axis(
    g3[2, 1],
    xticks=[tspan...], yticks=([ylims...], ["", ""]),
    title="EK1 (local&scalar)",
)
plot_errs!(ax, sol, ref; markersize=0, colors=COLORS)
xlims!(ax, tspan...)
ylims!(ax, ylims...)


rowgap!(g1, 2)
rowgap!(g2, 2)
rowgap!(g3, 2)

resize_to_layout!(fig)

path = joinpath(DIR, "..", "../figures/calibration.pdf")
save(path, fig)
@info "Saved figure to $path"
