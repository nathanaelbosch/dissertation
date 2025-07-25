using LinearAlgebra, Distributions, GaussianDistributions
using CairoMakie, TuePlots, LinearAlgebra, ColorSchemes, LaTeXStrings
using FiniteHorizonGramians
import KalmanFilterToolbox as KFT
using Random
Random.seed!(3)
using Dissertation

DIR = @__DIR__

COLORS = Makie.wong_colors()
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
        nrows=1, ncols=2.5,
    )
))

_plotrange = [0, 2*2pi]
N = 50
JITTER = 1e-12
CLIP = 1e-0

WP() = begin
    mu0, Sigma0 = [0.0], Matrix(1.0I, 1, 1)
    F, G, E = [0.0;;], [1.0;;], [1.0;;]
    return (mu0, Sigma0), (F, G, E)
end
WV() = begin
    mu0, Sigma0 = [0.0; 0.0], Matrix(1.0I, 2, 2)
    F, G, E = [0.0 1.0; 0.0 0.0], [0.0; 1.0;;], [1.0 0.0]
    return (mu0, Sigma0), (F, G, E)
end
gmp = WV()
times = range(_plotrange..., length=N+1)
plottimes = range(_plotrange..., length=250)
σ = 0.5
f(t) = sin(t)
data = [f(t) + σ * randn() for t in times]

# function sample(gmp; plotrange=range(_plotrange..., length=500), RNG=MersenneTwister(0))
#     (mu0, Sigma0), (F, G, E) = gmp
#     s = rand(RNG, Gaussian(mu0, Sigma0))
#     out = [E * s]
#     for i in 2:length(plotrange)
#         dt = plotrange[i] - plotrange[i-1]
#         A, Q = FiniteHorizonGramians.exp_and_gram(
#             F, G, dt, FiniteHorizonGramians.AdaptiveExpAndGram{Float64}())
#         s = rand(RNG, Gaussian(A * s, Q))
#         push!(out, E * s)
#     end
#     return out
# end

function get_posterior(gmp; plottimes=plottimes, times, data, update=true, smooth=true)
    _times = union(plottimes, times) |> unique |> sort
    (mu0, Sigma0), (F, G, E) = gmp
    mu = mu0
    Sigma = Sigma0
    out = [(mu, Sigma)]
    for i in 2:length(_times)
        t = _times[i]
        dt = _times[i] - _times[i-1]
        A, Q = FiniteHorizonGramians.exp_and_gram(
            F, G, dt, FiniteHorizonGramians.AdaptiveExpAndGram{Float64}())
        mu, Sigma = KFT.predict(mu, Sigma, A, zero(mu), Q)
        if update && t in times
            j = findfirst(isequal(t), times)
            mu, Sigma = KFT.update(mu, Sigma, [data[j]], E, [0], [σ^2;;])
        end
        push!(out, (mu, Sigma))
    end
    if smooth
        for i in length(_times)-1:-1:1
            dt = _times[i+1] - _times[i]
            A, Q = FiniteHorizonGramians.exp_and_gram(
                F, G, dt, FiniteHorizonGramians.AdaptiveExpAndGram{Float64}())
            mu, Sigma = KFT.smooth(out[i]..., out[i+1]..., A, zero(mu), Q)
            out[i] = (mu, Sigma)
        end
    end

    out = [((E * mu)[1], (E * Sigma * E')[1]) for (mu, Sigma) in out]
    return out
end
function plot_process!(ax, gmp; plottimes=plottimes, times=times, data=data, update=true, smooth=true, kwargs...)
    _times = union(plottimes, times) |> unique |> sort
    post = get_posterior(gmp; times, data, update, smooth)
    means = [p[1] for p in post]
    stddevs = [sqrt(p[2]) for p in post]
    lines!(ax, _times, means; kwargs...)
    band!(ax, _times, means-1.96stddevs, means+1.96stddevs;
          # color=(:gray, 0.5),
          kwargs...,
          color=(kwargs[:color], 0.3))
end
function plot_data!(ax, times=times, data=data, kwargs...)
    scatter!(ax, times[2:end], data[2:end]; markersize=3, color=:black, kwargs...)
end
function plot_truesignal!(ax, times=times, kwargs...)
    lines!(ax, times, [f(t) for t in times]; linestyle=:dash, kwargs...)
end


fig = Figure()
pax = Axis(
    fig[1, 1];
    title="Prior",
    yticks=[-5, 0, 5],
    xticks=([0, 4pi], ["0", "4π"]),
    xlabel="t",
    ylabel="y(t)",
)
plot_process!(pax, gmp; update=false, smooth=false, color=COLORS[1])
plot_truesignal!(pax)
plot_data!(pax)

fax = Axis(
    fig[1, 2];
    title="Filter output",
    yticks=[-2, 0, 2],
    xticks=([0, 4pi], ["0", "4π"]),
    xlabel="t",
)
plot_process!(fax, gmp; update=true, smooth=false, color=COLORS[2])
plot_truesignal!(fax)
plot_data!(fax)

sax = Axis(
    fig[1, 3];
    title="Posterior",
    yticks=[-2, 0, 2],
    xticks=([0, 4pi], ["0", "4π"]),
    xlabel="t",
)
plot_process!(sax, gmp; update=true, smooth=true, color=COLORS[3])
plot_truesignal!(sax)
plot_data!(sax)

# hidespines!(pax, :t, :r)
# hidespines!(fax, :t, :r)
# hidespines!(sax, :t, :r)
ylims!(pax, -5, 5)
ylims!(fax, -2, 2)
ylims!(sax, -2, 2)
xlims!(pax, _plotrange...)
xlims!(fax, _plotrange...)
xlims!(sax, _plotrange...)

path = joinpath(DIR, "..", "../figures/gmp_regression.pdf")
save(path, fig)
@info "Saved figure to $path"
