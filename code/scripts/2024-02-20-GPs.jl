using LinearAlgebra, Distributions, Random
using CairoMakie, TuePlots, LinearAlgebra, ColorSchemes, LaTeXStrings
using Dissertation

using KernelFunctions, PDMats, AbstractGPs

struct WienerVelocityKernel <: KernelFunctions.Kernel end
(::WienerVelocityKernel)(x, y) = (min(x, y)^3 / 3 + abs(x - y) * min(x, y)^2 / 2)

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
        nrows=1, ncols=2,
    )
))

_plotrange = [1e-10, 15]
N = 5
JITTER = 1e-12
CLIP = 1e-0

function plot_kernel!(ax, gp; plotrange=range(_plotrange..., length=200), kwargs...)
    K = cov(gp(plotrange, JITTER))
    K = max.(K, 0)
    # K = sqrt.(K)
    heatmap!(
        ax, plotrange, plotrange, K[:, end:-1:1];
        colorrange=(0, maximum(abs, K)),
        # colormap=:amp,
        colormap=Reverse(ColorSchemes.grays),
        # colormap=ColorSchemes.grays,
        kwargs...)
end

function plot_precision!(ax, gp; plotrange=range(_plotrange..., length=200), kwargs...)
    K = cov(gp(plotrange, JITTER))
    P = inv(K)
    V = min(maximum(abs, P), CLIP)
    heatmap!(
        ax, plotrange, plotrange, P[:, end:-1:1];
        colorrange=(-V, V),
        colormap=:bwr,
        kwargs...)
end

function plot_samples!(ax, gp; plotrange=range(_plotrange..., length=500), N=5, kwargs...)
    samples = rand(MersenneTwister(0), gp(plotrange, JITTER), N)
    for i in 1:N
        lines!(ax, plotrange, samples[:, i];
            alpha=0.5, kwargs...)
    end
end


fig = Figure()
contents = (
    "Wiener process" => WienerKernel(),
    "Wiener velocity" => WienerVelocityKernel(),
    "Matern 1/2" => Matern12Kernel(),
    "Matern 3/2" => Matern32Kernel(),
    # "Matern 5/2" => Matern52Kernel(),
    "Squared exponential" => SEKernel(),
    )
for (i, (title, kernel)) in enumerate(contents)
    gp = GP(kernel)
    sax = Axis(fig[1, i];
        title,
        ylabel=i == 1 ? "Samples" : "",
    )
    plot_samples!(sax, gp)
    hidedecorations!(sax; label=false)
    # hidespines!(sax, :l, :t, :r, :b)
    xlims!(sax, _plotrange...)
    # ylims!(sax, -5, 10)

    kax = Axis(fig[2, i];
        ylabel=i == 1 ? "Kernel\nmatrix" : "",
        aspect=DataAspect(),
    )
    plot_kernel!(kax, gp)
    hidedecorations!(kax; label=false)

    # pax = Axis(fig[3, i];
    #     ylabel=i == 1 ? "Precision\nmatrix" : "",
    #     aspect=DataAspect(),
    # )
    # plot_precision!(pax, gp)
    # hidedecorations!(pax; label=false)
end
# for i in 2:length(contents)
#     linkyaxes!(fig.content[1], fig.content[2*i-1])
# end

path = joinpath(DIR, "..", "../figures/gp_priors.pdf")
save(path, fig)
@info "Saved figure to $path"


############################################################################################
# Posteriors
############################################################################################
ts = [π/2, 1.8π, 3.5π]
ys = [0.5, -3, 3]
obsvar = 1e-10

fig = Figure()
for (i, (title, kernel)) in enumerate((
    "Wiener process" => WienerKernel(),
    "Wiener velocity" => WienerVelocityKernel(),
    "Matern 1/2" => Matern12Kernel(),
    "Matern 3/2" => Matern32Kernel(),
    # "Matern 5/2" => Matern52Kernel(),
    "Squared exponential" => SEKernel(),
))
    gp = posterior(GP(kernel)(ts, obsvar), ys)

    sax = Axis(fig[1, i];
        title,
        ylabel=i == 1 ? "Samples" : "",
    )
    plot_samples!(sax, gp)
    hidedecorations!(sax; label=false)
    # hidespines!(sax, :l, :t, :r, :b)
    scatter!(sax, ts, ys; color=:black, markersize=5)
    xlims!(sax, _plotrange...)
    ylims!(sax, -5, 5)

    kax = Axis(fig[2, i];
        ylabel=i == 1 ? "Kernel\nmatrix" : "",
        aspect=DataAspect(),
    )
    plot_kernel!(kax, gp)
    hidedecorations!(kax; label=false)

    # pax = Axis(fig[3, i];
    #     ylabel=i == 1 ? "Precision\nmatrix" : "",
    #     aspect=DataAspect(),
    # )
    # plot_precision!(pax, gp)
    # hidedecorations!(pax; label=false)
end

path = joinpath(DIR, "..", "../figures/gp_posteriors.pdf")
save(path, fig)
@info "Saved figure to $path"
