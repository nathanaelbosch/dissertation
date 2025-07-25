using ProbNumDiffEq, CairoMakie, TuePlots, LinearAlgebra, ColorSchemes, LaTeXStrings
using Dissertation

DIR = @__DIR__

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
        nrows=1, ncols=1,
    )
))

fig = Figure()

function plot_samples!(ax, prior; plotrange=range(0, 15, length=250), N=5, kwargs...)
    samples = ProbNumDiffEq.sample(prior, plotrange, N) |> stack
    samples = permutedims(samples, (3, 1, 2))

    d = ProbNumDiffEq.dim(prior)
    q = ProbNumDiffEq.num_derivatives(prior)
    E0 = reshape(ProbNumDiffEq.projection(d, q)(0), d * (q + 1))'

    for i in 1:N
        s = samples[:, :, i] * E0'
        lines!(ax, plotrange, s;
               # color = :black,
               palette=:viridis,
            alpha=0.5, kwargs...)
    end
end

plot_samples!(Axis(fig[1, 0]; title="IWP(0)"), IWP(0))
plot_samples!(Axis(fig[1, 1]; title="IWP(1)"), IWP(1))
plot_samples!(Axis(fig[1, 2]; title="IWP(2)"), IWP(2))
plot_samples!(Axis(fig[1, 3]; title="IWP(3)"), IWP(3))

plot_samples!(Axis(fig[2, 0]; title="IOUP(0, -2)"), IOUP(0, -2))
plot_samples!(Axis(fig[2, 1]; title="IOUP(1, -2)"), IOUP(1, -2))
plot_samples!(Axis(fig[2, 2]; title="IOUP(2, -2)"), IOUP(2, -2))
plot_samples!(Axis(fig[2, 3]; title="IOUP(3, -2)"), IOUP(3, -2))

plot_samples!(Axis(fig[3, 0]; title="Matern 1/2"), Matern(1//2, 1))
plot_samples!(Axis(fig[3, 1]; title="Matern 3/2"), Matern(3//2, 1))
plot_samples!(Axis(fig[3, 2]; title="Matern 5/2"), Matern(5//2, 1))
plot_samples!(Axis(fig[3, 3]; title="Matern 7/2"), Matern(7//2, 1))

for ax in fig.content
    hidedecorations!(ax)
    hidespines!(ax, :l, :t, :r, :b)
end

path = joinpath(DIR, "..", "../figures/priors.pdf")
save(path, fig)
@info "Saved figure to $path"
