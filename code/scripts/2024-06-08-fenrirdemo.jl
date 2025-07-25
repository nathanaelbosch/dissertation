using ProbNumDiffEq, LinearAlgebra, OrdinaryDiffEq#, Plots
using ProbNumDiffEq.DataLikelihoods
using Optimization, OptimizationOptimJL, OptimizationBBO
using Random
Random.seed!(1234)


function f(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1 / c) * (u[1] - a - b * u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
true_prob = ODEProblem(f, u0, tspan, p)

true_sol = solve(true_prob, Vern9(), abstol=1e-10, reltol=1e-10)

times = 1:0.1:20
σ = 1e-1
H = [1 0;]
odedata = [H * true_sol(t) .+ σ * randn() for t in times]

θ_est = [0.01, 0.01, 1.0]
u0_est = [0.0, 0.0]
prob = remake(true_prob, p=θ_est, u0=u0_est)


# test:
sol = solve(prob, EK1(), abstol=1e-8, reltol=1e-8)

data = (t=times, u=odedata)
nll = -fenrir_data_loglik(
    prob, EK1(smooth=true);
    data, observation_noise_cov=σ^2, observation_matrix=H,
    adaptive=false, dt=1e-1)

function loss(x, _)
    ode_params = x[1:3]
    u0_params = x[4:5]
    prob = remake(true_prob, p=ode_params, u0=u0_params)
    κ² = exp(x[end]) # we also optimize the diffusion parameter of the EK1
    # l = x[end-1]
    return -fenrir_data_loglik(
        prob, EK1(smooth=true,
                  # prior=Matern(3, l),
                  # initialization=SimpleInit(),
                  diffusionmodel=FixedDiffusion(κ², false));
        data, observation_noise_cov=σ^2, observation_matrix=H,
        adaptive=false, dt=1e-2
    )
end

fun = OptimizationFunction(loss, Optimization.AutoForwardDiff())
x0 = [θ_est..., u0_est..., 20]
optprob = OptimizationProblem(
    fun, x0;
    # lb=[0.0, 0.0, 0.0, -10, -10, -50], ub=[1.0, 1.0, 8, 10, 10, 100] # lower and upper bounds
    lb=[0.0, 0.0, 0.0, -10, -10, -50], ub=[1.0, 1.0, 20, 10, 10, 100] # lower and upper bounds
)

param_trajectory = typeof(x0)[]
nll_trajectory = Float64[]
_cb = function (state, args...; kwargs...)
    @info "[i=$(state.iter)]" state.u state.objective
    push!(param_trajectory, state.u)
    push!(nll_trajectory, state.objective)
    return false
end
optsol = solve(optprob, LBFGS(); callback=_cb)
# optsol = solve(optprob, LBFGS(linesearch=OptimizationOptimJL.Optim.BackTracking()); callback=_cb)
# optsol = solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(); callback=_cb)
p_mle = optsol.u[1:3]


# function rkloss(x, _)
#     ode_params = x[1:3]
#     u0_params = x[4:5]
#     prob = remake(true_prob, p=ode_params, u0=u0_params)
#     sol = solve(prob, Rosenbrock23(), saveat=data.t, sensealg=DiffEqBase.SensitivityADPassThrough())
#     loss = sum(sum.(abs2, data.u .- Ref(H) .* sol.u))
#     return loss, sol
# end
# rkfun = OptimizationFunction(rkloss, Optimization.AutoForwardDiff())
# rkoptprob = OptimizationProblem(
#     rkfun, [θ_est..., u0_est...];
#     lb=[0.0, 0.0, 0.0, -10, -10], ub=[1.0, 1.0, 8.0, 10, 10])
# rkparam_trajectory = typeof(x0)[]
# rknll_trajectory = Float64[]
# rk_cb = function (state, args...; kwargs...)
#     @info "[i=$(state.iter)]" state.u state.objective
#     push!(rkparam_trajectory, state.u)
#     push!(rknll_trajectory, state.objective)
#     return false
# end
# rkoptsol = solve(rkoptprob, LBFGS(); callback=rk_cb)
# rkoptsol = solve(rkoptprob, LBFGS(linesearch=OptimizationOptimJL.Optim.BackTracking()); callback=_cb)
# rkp_mle = rkoptsol.u[1:3]


##############################################################
DIR = @__DIR__

using OrdinaryDiffEq, ProbNumDiffEq
using CairoMakie, TuePlots, ColorSchemes
using Dissertation
using Statistics

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

fig = Figure()
ax1 = Axis(
    fig[1, 1],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Initial trajectory",
    xlabel="t",
    ylabel="y(t)",
)
plot_sol!(ax1, true_sol; markersize=0, colors=(:black, :black), linestyle=:dash, label="true solution")
plot_sol!(ax1, sol; markersize=0, colors=COLORS, label="initial parameters")
scatter!(ax1, times, stack(odedata)[:], color=:gray, markersize=4, label="observations")
# axislegend(ax1)
ylims!(ax1, -2.5, 2.5)


ax2 = Axis(
    fig[1, 2],
    # yticks=[0, 1],
    xticks=[tspan...],
    title="Optimized trajectory",
    xlabel="t",
    ylabel="y(t)",
)
p_mle = optsol.u[1:3]
sol2 = solve(remake(true_prob, p=p_mle), EK1())
plot_sol!(ax2, true_sol; markersize=0, colors=(:black, :black), linestyle=:dash, label="true solution")
plot_sol!(ax2, sol2; markersize=0, colors=COLORS, label="optimized trajectory")
# rksol2 = solve(remake(true_prob, p=rkp_mle), Rosenbrock23())
# plot_sol!(ax2, rksol2; markersize=0, colors=COLORS, label="optimized trajectory", linestyle=:dash)
scatter!(ax2, times, stack(odedata)[:], color=:gray, markersize=4, label="observations")
# axislegend(ax2)
ylims!(ax2, -2.5, 2.5)


labels = ["α", "β", "γ", "y₁", "y₂", "log(σ²)"]
gl = fig[1, 3] = GridLayout()
_p = [p..., u0...]
for i in 1:length(param_trajectory[1])
    ax = Axis(
        gl[
            (i-1)÷3+1,
            (i-1)%3,
        ],
        # title=i==1 ? "Parameter optmization paths" : "",
        # xlabel="t",
        # ylabel=labels[i],
        title=labels[i],
        # xticklabelsvisible=i==length(param_trajectory[1]),
        xticklabelsvisible=((i - 1) ÷ 3 == 1),
        xticks=0:200:length(param_trajectory),
    )
    if i <= length(_p)
        hlines!(ax, [_p[i]], color=:black, linestyle=:dash, linewidth=0.5)
    end
    lines!(ax, 1:length(param_trajectory), [p[i] for p in param_trajectory])
end


rowgap!(gl, 2)
colgap!(gl, 2)

# Add legend to the bottom
Legend(
    fig[2, 1:2],
    [LineElement(color=:black, linestyle=:dash, linewidth=1),
        MarkerElement(color=:gray, marker=:o, markersize=4),
        [LineElement(color=COLORS[1],
                linewidth=1,
                points=Point2f[(0, 0.66), (1, 0.66)]),
            LineElement(color=COLORS[2],
                linewidth=1,
                points=Point2f[(0, 0.33), (1, 0.33)])],
    ],
    ["True trajectory",
        "Observations",
        "Estimated trajectory",
    ],
    tellheight=true,
    orientation=:horizontal,
    patchsize=(10, 10),
    colgap=10,
)
rowgap!(fig.layout, 2)

# path = joinpath(DIR, "..", "../figures/fenrirfail.pdf")
path = joinpath(DIR, "..", "../figures/fenrirdemo.pdf")
save(path, fig)
@info "Saved figure to $path"
