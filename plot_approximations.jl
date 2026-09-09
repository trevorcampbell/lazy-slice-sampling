using Random
using Plots
using LaTeXStrings

include("cost_formulae.jl")

function main()
	default(
    	titlefontsize = 18,
    	guidefontsize = 16,
    	tickfontsize = 14,
    	legendfontsize = 20,
    	linewidth=2,
    	linestyle=:solid,
    	ylims=:auto,
    	color=:black,
    	xscale=:log10,
    	yscale=:identity,
    	xlabel="w/λ",
 		ylabel="Expected Cost",
 		dpi=200
	)


	ws = 10.0 .^(-3.01:0.001:3.01)

	# compare stepping cost to its approximation
	p = plot(ws, step_cost_total.(ws), label=nothing, ylims=(0, 75), xticks= 10.0 .^(-3:3))
	plot!(ws, step_cost_approximation_separable.(ws, 1.0), label=nothing, linestyle=:dash)
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:solid, label=L"f_{\mathrm{step}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dash, label=L"\hat{f}_{\mathrm{step}}")
	savefig(p, "step_approximation.png")
	display(p)
	readline()

	# compare doubling cost to its approximations
	p = plot(ws, double_cost_total.(ws), label=nothing, legend=:topright, xticks= 10.0 .^(-3:3))
	plot!(ws, double_cost_approximation_separable.(ws, 1.0), label=nothing, linestyle=:dash)
	plot!(ws, double_cost_approximation_convex.(ws, 1.0), label=nothing, linestyle=:dot)
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:solid, label=L"f_{\mathrm{cache}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dash, label=L"\hat{f}_{\mathrm{cache}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dot, label=L"\tilde{f}_{\mathrm{cache}}")
	savefig(p, "double_approximation.png")
	display(p)
	readline()

	p = plot(ws, lazydouble_cost_total.(ws), label=nothing, legend=:bottomright, xticks= 10.0 .^(-3:3))
	plot!(ws, lazydouble_cost_approximation_separable.(ws, 1.0), label=nothing, linestyle=:dash)
	plot!(ws, lazydouble_cost_approximation_convex.(ws, 1.0), label=nothing, linestyle=:dot)
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:solid, label=L"f_{\mathrm{lazy}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dash, label=L"\hat{f}_{\mathrm{lazy}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dot, label=L"\tilde{f}_{\mathrm{lazy}}")
	savefig(p, "lazy_approximation.png")
	display(p)
	readline()

	# plot relative (eps,delta) error curves
	ws = 10.0 .^(-6:0.0001:6)
	relerr_step = abs.(step_cost_approximation_separable.(ws, 1.0) - step_cost_total.(ws))./(0.5*step_cost_total.(ws))
	relerr_double_sep = abs.(double_cost_approximation_separable.(ws, 1.0) - double_cost_total.(ws))./(2.0)
	relerr_double_cvx = abs.(double_cost_approximation_convex.(ws, 1.0) - double_cost_total.(ws))./(0.35)
	relerr_lazydouble_sep = abs.(lazydouble_cost_approximation_separable.(ws, 1.0) - lazydouble_cost_total.(ws))./(2.0)
	relerr_lazydouble_cvx = abs.(lazydouble_cost_approximation_convex.(ws, 1.0) - lazydouble_cost_total.(ws))./(0.35)
	
	p = plot(ws, relerr_step, label=nothing, ylabel="(ϵ,δ) Relative Error", legend=:bottomleft, yscale=:identity)
	plot!([NaN], [NaN], linewidth=1, label=L"\hat{f}_{\mathrm{step}}")
	hline!([1.0], color=:grey, linestyle=:dash, label=nothing)
	savefig(p, "approximation_relative_errors_step.png")
	display(p)
	readline()

	p = plot(ws, relerr_double_sep, label=nothing,ylabel="(ϵ,δ) Relative Error", legend=:bottomleft,  linestyle=:solid)
	plot!(ws, relerr_double_cvx, label=nothing, linestyle=:dash)
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:solid, label=L"\hat{f}_{\mathrm{cache}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dash, label=L"\tilde{f}_{\mathrm{cache}}")
	hline!([1.0], color=:grey, linestyle=:dash, label=nothing)
	savefig(p, "approximation_relative_errors_cache.png")
	display(p)
	readline()

	p = plot(ws, relerr_lazydouble_sep, label=nothing,ylabel="(ϵ,δ) Relative Error", legend=:topleft, linestyle=:solid)
	plot!(ws, relerr_lazydouble_cvx, label=nothing, linestyle=:dash)
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:solid, label=L"\hat{f}_{\mathrm{lazy}}")
	plot!([NaN], [NaN], linewidth=0.7, linestyle=:dash, label=L"\tilde{f}_{\mathrm{lazy}}")
	hline!([1.0], color=:grey, linestyle=:dash, label=nothing)
	savefig(p, "approximation_relative_errors_lazy.png")
	display(p)
	readline()
end

main()

