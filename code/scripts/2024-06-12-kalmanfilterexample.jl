DIR = @__DIR__

using LinearAlgebra, Random, GaussianDistributions, Statistics
import KalmanFilterToolbox as KFT
using CairoMakie, TuePlots, ColorSchemes
using Dissertation

COLORS = ColorSchemes.tableau_10.colors[[1, 2, 3, 4, 5]]
COLORS = Makie.wong_colors()
datacolor = :black
set_theme!(merge(
    Theme(
        palette=(
            color=Makie.wong_colors(),
            # color=ColorSchemes.grays1,
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

fig = Figure()
ax1 = Axis(fig[1, 1]; title="Problem setting",
    xticks=([0, 4pi], ["0", "4π"]))
ax2 = Axis(fig[1, 2]; title="Filtering",
    xticks=([0, 4pi], ["0", "4π"]),
    yticklabelsvisible=false)
ax3 = Axis(fig[1, 3]; title="Smoothing",
    xticks=([0, 4pi], ["0", "4π"]),
    yticklabelsvisible=false)
ax4 = Axis(fig[1, 4]; title="Posterior samples",
    xticks=([0, 4pi], ["0", "4π"]),
    yticklabelsvisible=false)
linkyaxes!(ax1, ax2)
linkyaxes!(ax1, ax3)
linkyaxes!(ax1, ax4)

N = 100
ts = range(0, 4π, length=N+1)[1:end]
dt = 4π/N
f(t) = sin(t)
_R = 0.5^2
data = f.(ts) .+ sqrt(_R) .* randn(length(ts))
lines!(ax1, ts[1]:0.1:ts[end], f, color=:black, linestyle=:dash)
scatter!(ax1, ts, data, markersize=3, color=datacolor)

function filter(prior, data; smooth=false)
    d, q = prior.wiener_process_dimension, prior.num_derivatives
    D = d * (q + 1)
    E0 = KFT.projectionmatrix(d, q, 0)

    A, Q = KFT.discretize(prior, dt)
    b = zeros(D)
    R, v = Matrix(_R*I, d, d), zeros(d)
    m, C = zeros(D), Matrix(1.0I, D, D)

    N = length(data)
    xs = [Gaussian(m, C)]
    backward_transitions = []
    for i in 1:N
        mp, Cp = KFT.predict(m, C, A, b, Q)
        if smooth
            push!(backward_transitions, KFT.get_backward_transition(m, C, mp, Cp, A))
        end
        m, C = KFT.update(mp, Cp, [data[i]], E0, v, R)
        push!(xs, Gaussian(m, C))

    end

    if smooth
        m, C = mean(xs[end]), cov(xs[end])
        @assert length(xs) == N + 1
        for i in N:-1:1
            A, b, Q = backward_transitions[i]
            m, C = KFT.predict(m, C, A, b, Q)
            xs[i] = Gaussian(m, C)
        end
    end

    return xs, backward_transitions
end
d, q = 1, 1
prior = KFT.IWP(d, q)
xs, _ = filter(prior, data)
E0 = KFT.projectionmatrix(d, q, 0)
ms = map(x -> (E0*mean(x))[1], xs) |> stack
stds = map(x -> (sqrt.(diag(E0 * cov(x) * E0')))[1], xs) |> stack

lines!(ax2, 0:0.1:100, f, color=:black, linestyle=:dash)
scatter!(ax2, ts, data, markersize=3, color=datacolor)
lines!(ax2, [0,ts...], ms, color=COLORS[2])
fill_between!(ax2, [0, ts...], ms - 1.96stds, ms + 1.96stds, color=(COLORS[2], 0.3))


xs, bts = filter(prior, data; smooth=true)
ms = map(x -> (E0*mean(x))[1], xs) |> stack
stds = map(x -> (sqrt.(diag(E0 * cov(x) * E0')))[1], xs) |> stack

lines!(ax3, 0:0.1:100, f, color=:black, linestyle=:dash)
scatter!(ax3, ts, data, markersize=3, color=datacolor)
lines!(ax3, [0, ts...], ms, color=COLORS[3])
fill_between!(ax3, [0, ts...], ms - 1.96stds, ms + 1.96stds, color=(COLORS[3], 0.3))

_rand(x) = rand(Gaussian(x.μ, Symmetric(x.Σ + 0.00001I)))

function sample_backwards(bts, xfinal)
    x = _rand(xfinal)
    E0 = KFT.projectionmatrix(d, q, 0)
    out = [(E0*x)[1]]
    for bt in reverse(bts)
        A, b, Q = bt
        x = _rand(Gaussian(A * x + b, Q))
        push!(out, (E0*x)[1])
    end
    return reverse(out)
end
lines!(ax4, 0:0.1:100, f, color=:black, linestyle=:dash)
scatter!(ax4, ts, data, markersize=3, color=datacolor)
for _ in 1:10
    s = sample_backwards(bts, xs[end])
    lines!(ax4, [0, ts...], s, linewidth=0.5, color=(COLORS[4], 0.8))
end



xlims!(ax1, ts[1], ts[end])
xlims!(ax2, ts[1], ts[end])
xlims!(ax3, ts[1], ts[end])
xlims!(ax4, ts[1], ts[end])


path = joinpath(DIR, "..", "../figures/kalmanfilterexample.pdf")
save(path, fig)
@info "Saved figure to $path"
