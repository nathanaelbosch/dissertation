DIR = @__DIR__

using LinearAlgebra, Random, GaussianDistributions, Statistics, Distributions
import KalmanFilterToolbox as KFT
using CairoMakie, TuePlots, ColorSchemes
using Dissertation
using Random
Random.seed!(1337)

COLORS = ColorSchemes.tableau_10.colors[[1, 2, 3, 4, 5]]
COLORS = Makie.wong_colors()
set_theme!(merge(
    Theme(
        palette=(
            color=Makie.wong_colors(),
            # color=ColorSchemes.grays1,
        ),
        Axis=(
            Scatter=(;
                markersize=1,
                # strokewidth=0.1,
            ),
        )
    ),
    DissertationTheme,
    Theme(
        TuePlots.SETTINGS[:NEURIPS];
        font=false,
        fontsize=true,
        figsize=true,
        thinned=true,
        nrows=2, ncols=4,
    ),
))

fig = Figure()
ax1 = Axis(fig[1, 1]; title="Problem setting")
# ax2 = Axis(fig[1, 2]; title="KF/KS",
#     yticklabelsvisible=false)
ax3 = Axis(fig[1, 2]; title="EKS",
    yticklabelsvisible=false)
ax4 = Axis(fig[1, 3]; title="IEKS",
    yticklabelsvisible=false)
# linkyaxes!(ax1, ax2)
linkyaxes!(ax1, ax3)
linkyaxes!(ax1, ax4)

############################################################################################
# The model
dt = 0.0125
# dt = 0.05
g = 9.81
q_c = 1
r = 0.3
f(x) = [x[1] + x[2] * dt; x[2] - g * sin(x[1]) * dt]
Q = q_c * [dt^3/3 dt^2/2; dt^2/2 dt] |> Symmetric
_h(x) = sin(x[1])
h(x) = [_h(x)]
_R = r^2
R = [R;;]
x0 = [pi / 2; 0]

############################################################################################
# Simulate pendulum and get measurements
ts = 0:dt:5
xtrues = [x0]
for t in ts[2:end]
    x = rand(Gaussian(f(xtrues[end]), Q))
    push!(xtrues, x)
end
P = Ref([1; 0]')
lines!(ax1, ts, P .* xtrues, color=:black, linestyle=:dash)

data = _h.(xtrues) .+ sqrt(_R) .* randn(length(ts))
scatter!(ax1, ts, data, markersize=2)

# dt = 1//10


function ekf(; smooth=false, iters=1)
    backward_transitions = []
    N = length(data)
    m, C = x0, Matrix(1e0I, 2, 2)
    m, C = KFT.update(m, C, [data[1]], KFT.linearize(h, m)..., R)
    xs = [Gaussian(m, C)]

    for i in 2:N
        A, b = KFT.linearize(f, m)
        mp, Cp = KFT.predict(m, C, A, b, Q)

        push!(backward_transitions, KFT.get_backward_transition(m, C, mp, Cp, A))

        m, C = KFT.update(mp, Cp, [data[i]], KFT.linearize(h, mp)..., R)
        push!(xs, Gaussian(m, C))

    end

    if smooth
        m, C = mean(xs[end]), cov(xs[end])
        for i in N-1:-1:1
            A, b, Q = backward_transitions[i]
            m, C = KFT.predict(m, C, A, b, Q)
            xs[i] = Gaussian(m, C)
        end
    end

    return xs, backward_transitions
end
xs, bts = ekf(; smooth=true)
ms = mean.(P .* xs)
stds = std.(P .* xs)
lines!(ax3, ts, P .* xtrues, color=:black, linestyle=:dash)
scatter!(ax3, ts, data, markersize=2)
lines!(ax3, ts, ms, color=COLORS[3])
fill_between!(ax3, ts, ms - 1.96stds, ms + 1.96stds, color=(COLORS[3], 0.3))


function ieks(; smooth=true, iters=1)
    N = length(data)

    _xs, _ = ekf(; smooth=true)
    xi = copy(mean.(_xs))
    @assert length(xi) == N

    xs = []
    backward_transitions = []
    for _ in 1:iters
        backward_transitions = []
        m, C = x0, Matrix(1e-6I, 2, 2)
        m, C = KFT.update(m, C, [data[1]], KFT.linearize(h, xi[1])..., R)
        xs = [Gaussian(m, C)]

        for i in 2:N
            A, b = KFT.linearize(f, xi[i-1])
            mp, Cp = KFT.predict(m, C, A, b, Q)

            push!(backward_transitions, KFT.get_backward_transition(m, C, mp, Cp, A))

            m, C = KFT.update(mp, Cp, [data[i]], KFT.linearize(h, xi[i])..., R)
            push!(xs, Gaussian(m, C))

        end

        if smooth
            m, C = mean(xs[end]), cov(xs[end])
            for i in N-1:-1:1
                A, b, Q = backward_transitions[i]
                m, C = KFT.predict(m, C, A, b, Q)
                xs[i] = Gaussian(m, C)
            end
        end
        xi = copy(mean.(xs))
    end

    return xs, backward_transitions
end
xs_ieks, bts = ieks(; iters=1000)
ms_ieks = mean.(P .* xs_ieks)
stds_ieks = std.(P .* xs_ieks)
lines!(ax4, ts, P .* xtrues, color=:black, linestyle=:dash)
scatter!(ax4, ts, data, markersize=2)
lines!(ax4, ts, ms_ieks, color=COLORS[2])
fill_between!(ax4, ts, ms_ieks - 1.96stds_ieks, ms_ieks + 1.96stds_ieks, color=(COLORS[2], 0.3))


rmse(xs) = sqrt(mean(abs2, norm.(xtrues .- mean.(xs))))
@info "" eks_rmse = rmse(xs) ieks_rmse = rmse(xs_ieks)

function chisq(xs)
    diff = mean.(xs) .- xtrues
    N = length(diff) * length(diff[1])
    chi = sum([diff[i]' * (cov(xs[i]) \ diff[i]) for i in 1:length(diff)])
    return logpdf(Chisq(N), chi)
end
@info "" eks_chisq = chisq(xs) ieks_chisq = chisq(xs_ieks)

function likelihood(xs, bts)
    ll = logpdf(Gaussian(mean(xs[end]), Symmetric(cov(xs[end]))), xtrues[end])
    for i in (length(xtrues)-1):-1:1
        A, b, C = bts[i]
        ll += logpdf(Gaussian(A * xtrues[i+1] + b, Symmetric(C)), xtrues[i])
    end
    return ll
end
@info "" eks_ll = likelihood(xs, bts) ieks_ll = likelihood(xs_ieks, bts)



xlims!(ax1, 0, 5)
# xlims!(ax2, 0, 5)
xlims!(ax3, 0, 5)
xlims!(ax4, 0, 5)

path = joinpath(DIR, "..", "../figures/nlgssmexample.pdf")
save(path, fig)
@info "Saved figure to $path"
