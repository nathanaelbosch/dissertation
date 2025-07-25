module Dissertation

using LinearAlgebra, Statistics
using CairoMakie, TuePlots, ColorSchemes
using ProbNumDiffEq

include("plot_theme.jl")
export DissertationTheme



function plot_sol!(
    ax, sol;
    markersize=3,
    strokewidth=0.1,
    dense=false,
    colors=nothing,
    kwargs...
)
    d = length(sol.u[1])
    if !dense
        for i in 1:d
            if !isnothing(colors)
                kwargs = merge(kwargs, Dict(:color => colors[i]))
            end
            scatterlines!(
                ax, sol.t, stack(sol.u)[i, :];
                markersize,
                strokewidth,
                kwargs...
            )
        end
        if sol isa ProbNumDiffEq.ProbODESolution
            for i in 1:d
                band!(
                    ax, sol.t,
                    [mean(u)[i] - 1.96std(u)[i] for u in sol.pu],
                    [mean(u)[i] + 1.96std(u)[i] for u in sol.pu];
                    markersize,
                    strokewidth,
                    label=L"y_$i",
                    kwargs...,
                    color=colors==nothing ? (:gray, 0.4) : (colors[i], 0.3),
                )
            end
        end
    else
        error("Not yet implemented")
    end
end


function plot_errs!(ax, sol, ref; markersize=3, strokewidth=0.1, colors, logscale=false, kwargs...)
    d = length(sol.u[1])
    plotrange = range(sol.t[begin], sol.t[end], length=200)
    _sol = sol(plotrange)
    _ref = ref(plotrange)
    ref = ref(sol.t)
    if !logscale
        for i in 1:d
            band!(
                ax, _sol.t,
                [-1.96std(u)[i] for u in _sol.u],
                [1.96std(u)[i] for u in _sol.u];
                color=(colors[i], 0.4),
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
        for i in 1:d
            lines!(
                ax, _sol.t, [mean(u)[i] - r[i] for (u, r) in zip(_sol.u, _ref.u)];
                color=colors[i],
                label=L"y_$i",
                kwargs...)
            scatter!(
                ax, sol.t, [u[i] - r[i] for (u, r) in zip(sol.u, ref)];
                color=colors[i],
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
    else
        for i in 1:d
            lines!(
                ax, _sol.t[begin+1:end],
                [abs(mean(u)[i] - r[i]) for (u, r) in zip(_sol.u, _ref.u)][begin+1:end];
                color=colors[i],
                label=L"y_$i",
                kwargs...)
            scatter!(
                ax, sol.t[begin+1:end],
                [abs(u[i] - r[i]) for (u, r) in zip(sol.u, ref)][begin+1:end];
                color=colors[i],
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
            lines!(
                ax, _sol.t[begin+1:end],
                [1.96std(u)[i] for u in _sol.u][begin+1:end];
                color=(colors[i], 0.8),
                markersize,
                strokewidth,
                label=L"y_$i",
                linestyle=:dash,
                linewidth=0.8,
                kwargs...)
            band!(
                ax, _sol.t,
                [1e-20 for u in _sol.u],
                [1.96std(u)[i] for u in _sol.u];
                color=(colors[i], 0.4),
                markersize,
                strokewidth,
                label=L"y_$i",
                kwargs...)
        end
    end
end


end # module Dissertation
