using LinearAlgebra, Distributions, GaussianDistributions
using CairoMakie, TuePlots, LinearAlgebra, ColorSchemes, LaTeXStrings
using FiniteHorizonGramians
using Random
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
        nrows=0.5, ncols=2,
    )
))

_plotrange = [0, 15]
N = 5
JITTER = 1e-12
CLIP = 1e-0

WP() = begin
    mu0, Sigma0 = [0.0], [1e-20;;]
    F, G, E = [0.0;;], [1.0;;], [1.0;]'
    return (mu0, Sigma0), (F, G, E)
end
WV() = begin
    mu0, Sigma0 = [0.0; 0.0], 1e-10I(2)
    F, G, E = [0.0 1.0; 0.0 0.0], [0.0;1.0;;], [1.0;0.0]'
    return (mu0, Sigma0), (F, G, E)
end
Matern(s; l=1.0) = begin
    mu0, Sigma0 = zeros(s), 1e-10I(s)
    F = diagm(1 => ones(s-1))
    nu = s - 1//2
    lambda = sqrt(2nu) / l
    for i in 1:s
        ai = binomial(s, i-1)
        F[end, i] = - ai * lambda^(s-(i-1))
    end
    G = zeros(s, 1)
    G[end] = 1
    E = zeros(s)'
    E[1] = 1
    return (mu0, Sigma0), (F, G, E)
end

function sample(gmp; plotrange=range(_plotrange..., length=500), RNG=MersenneTwister(0))
    (mu0, Sigma0), (F, G, E) = gmp
    s = rand(RNG, Gaussian(mu0, Sigma0))
    out = [E * s]
    for i in 2:length(plotrange)
        dt = plotrange[i] - plotrange[i-1]
        A, Q = FiniteHorizonGramians.exp_and_gram(
            F, G, dt, FiniteHorizonGramians.AdaptiveExpAndGram{Float64}())
        s = rand(RNG, Gaussian(A * s, Q))
        push!(out, E * s)
    end
    return out
end

function plot_samples!(ax, gmp; plotrange=range(_plotrange..., length=500), N=5, kwargs...)
    for i in 1:N
        s = sample(gmp; plotrange, RNG=MersenneTwister(i+20))
        lines!(ax, plotrange, s; kwargs...)
    end
end


fig = Figure()
for (i, (title, gmp)) in enumerate((
    "Wiener process" => WP(),
    "Wiener velocity" => WV(),
    "Matern 1/2" => Matern(1),
    "Matern 3/2" => Matern(2),
    "Matern 5/2" => Matern(3),
    ))
    sax = Axis(fig[1, i];
        title,
        ylabel=i == 1 ? "Samples" : "",
    )
    plot_samples!(sax, gmp)
    hidedecorations!(sax; label=false)
    # hidespines!(sax, :l, :t, :r, :b)
    xlims!(sax, _plotrange...)
end

path = joinpath(DIR, "..", "../figures/gmp_priors.pdf")
save(path, fig)
@info "Saved figure to $path"
