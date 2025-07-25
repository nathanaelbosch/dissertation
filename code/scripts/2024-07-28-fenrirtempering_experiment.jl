using ProbNumDiffEq, LinearAlgebra, OrdinaryDiffEq#, Plots
using ProbNumDiffEq.DataLikelihoods
using Optimization, OptimizationOptimJL, OptimizationBBO
using Random
import Plots
Random.seed!(1234)


function f(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1 / c) * (u[1] - a - b * u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 40.0)
p = (0.2, 0.2, 3.0)
true_prob = ODEProblem(f, u0, tspan, p)

true_sol = solve(true_prob, Vern9(), abstol=1e-10, reltol=1e-10)

times = 1:0.1:tspan[2]
σ = 1e-1
H = [1 0;]
odedata = [H * true_sol(t) .+ σ * randn() for t in times]

θ_est = [0.0001, 0.0001, 19.0]
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
_cb = function (state, args...; display=true, kwargs...)
    @info "[i=$(state.iter)]" state.u state.objective
    push!(param_trajectory, state.u)
    push!(nll_trajectory, state.objective)
    if display==true
        ode_params = state.u[1:3]
        u0_params = state.u[4:5]
        κ² = exp(state.u[end]) # we also optimize the diffusion parameter of the EK1
        prob = remake(true_prob, p=ode_params, u0=u0_params)
        sol = solve(prob, EK1(smooth=true, diffusionmodel=FixedDiffusion(κ², false)),
                    adaptive=false, dt=1e-2)
        p = Plots.plot(sol; denseplot=false, label="")
        Plots.plot!(p, true_sol; denseplot=false, color=:black, linestyle=:dash, label="")
        Plots.scatter!(p, times, stack(odedata)[:]; color=:gray, markersize=5, label="")
        Plots.plot!(p; ylims=(-3, 3))
        Plots.display(p)
    end
    return false
end
optsol = solve(optprob,
               # LBFGS()
               LBFGS(linesearch=Optim.LineSearches.BackTracking())
               ; callback=_cb)
# optsol = solve(optprob, LBFGS(linesearch=OptimizationOptimJL.Optim.BackTracking()); callback=_cb)
# optsol = solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(); callback=_cb)
p_mle = optsol.u[1:3]


##############################################################
# Tempering
function loss2(x, κ²)
    ode_params = x[1:3]
    u0_params = x[4:5]
    prob = remake(true_prob, p=ode_params, u0=u0_params)
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
fun2 = OptimizationFunction(loss2, Optimization.AutoForwardDiff())
x0 = [θ_est..., u0_est...]
param_trajectory2 = typeof(x0)[]
nll_trajectory2 = Float64[]
optsol2 = nothing
for κ² in exp.(40:-1:0)
    @info "New diffusion" log(κ²)
    _cb2 = function (state, args...; display=true, kwargs...)
        @info "[i=$(state.iter)]" state.u state.objective log(κ²)
        push!(param_trajectory2, [state.u..., κ²])
        push!(nll_trajectory2, state.objective)
        if display == true
            ode_params = state.u[1:3]
            u0_params = state.u[4:5]
            prob = remake(true_prob, p=ode_params, u0=u0_params)
            sol = solve(prob, EK1(smooth=true, diffusionmodel=FixedDiffusion(κ², false)),
                adaptive=false, dt=1e-2)
            p = Plots.plot(sol; denseplot=false, label="")
            Plots.plot!(p, true_sol; denseplot=false, color=:black, linestyle=:dash, label="")
            Plots.scatter!(p, times, stack(odedata)[:]; color=:gray, markersize=1, label="")
            Plots.plot!(p; ylims=(-3, 3))
            Plots.display(p)
        end
        return state.iter > 500
        # return false # to try next
    end

    optprob2 = OptimizationProblem(
        fun2, x0, κ²;
        lb=[0.0, 0.0, 0.0, -10, -10], ub=[1.0, 1.0, 20, 10, 10] # lower and upper bounds
    )
    optsol2 = solve(optprob2,
        # LBFGS()
        LBFGS(linesearch=Optim.LineSearches.BackTracking())
        ; callback=_cb2)
    x0 = optsol2.u
    # push!(param_trajectory2, [optsol2.u..., κ²])
    # push!(nll_trajectory2, optsol2.objective)
end
optsol2
param_trajectory2
nll_trajectory2


optsolu = optsol.u
optsol2u = optsol2.u

appxsol = solve(remake(true_prob, p=p_mle, u0=u0_mle),
                EK1(smooth=true,
                    diffusionmodel=FixedDiffusion(exp(optsolu[end]), false)),
                adaptive=false, dt=1e-2)
appxsol2 = solve(remake(true_prob, p=p_mle2, u0=u0_mle2),
    EK1(smooth=true,
        diffusionmodel=FixedDiffusion(1.0, false)),
    adaptive=false, dt=1e-2)

using JLD2
jldsave("scripts/2024-07-28-fenrirtempering_data.jld2";
    tspan,
    prob, true_prob,
    sol, odedata, times,
    true_sol,
    appxsol, appxsol2,
    optsolu,
    optsol2u,
    param_trajectory,
    param_trajectory2,
    nll_trajectory,
    nll_trajectory2,
        p,
        u0,
        )
