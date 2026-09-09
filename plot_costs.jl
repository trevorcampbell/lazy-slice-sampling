using Random
using Plots

include("slice_sampling.jl")
include("cost_formulae.jl")
include("targets.jl")

function main()
	T = 1_000
	ws = 10.0 .^(-3:0.05:3)
	step_costs_w = zeros(length(ws),3)
	double_costs_w = zeros(length(ws),3)
	lazydouble_costs_w = zeros(length(ws),3)
	
	for i=1:length(ws)
		println("Iteration $i out of $(length(ws))")
		w = ws[i]
		costs_stepping = zeros(T,3)
		costs_doubling = zeros(T,3)
		costs_lazydoubling = zeros(T,3)
		#costs_hybrid = zeros(T,3)
		for t=1:T
			# draw from the target
			x = sample_uniform()
			fx = logpdf_uniform(x)
			# do one stepping out slice sampling move
			xp, fp, cost_slice, cost_shrink, cost_accept = slice_sample(x, ConstantW(w), logpdf_uniform, fx, :stepping, :default)
			costs_stepping[t,:] .= [cost_slice, cost_shrink, cost_accept]
			# do one doubling slice sampling move
			xp, fp, cost_slice, cost_shrink, cost_accept = slice_sample(x, ConstantW(w), logpdf_uniform, fx, :doubling, :default)
			costs_doubling[t,:] .= [cost_slice, cost_shrink, cost_accept]
			_, _, cost_slice, cost_shrink, cost_accept = slice_sample(x, ConstantW(w), logpdf_uniform, fx, :lazydoubling, :default)
			costs_lazydoubling[t,:] .= [cost_slice, cost_shrink, cost_accept]
		end
		step_costs_w[i,:] = sum(costs_stepping, dims=1)/T
		double_costs_w[i,:] = sum(costs_doubling, dims=1)/T
		lazydouble_costs_w[i,:] = sum(costs_lazydoubling, dims=1)/T
	end

	default(
    titlefontsize = 18,
    guidefontsize = 16,
    tickfontsize = 14,
    legendfontsize = 14,
    linewidth=2,
    label=nothing,
    xscale=:log10,
    yscale=:identity,
    xlabel="w/λ",
    ylabel="Expected Cost",
 	dpi=200,
 	color=:black,
 	markersize=6
	)

	println("Stepping")

	p = plot(ws, step_slice_cost.(ws, 1.0), label=nothing, linestyle=:dot, yscale=:log10, yticks= 10.0 .^[0,1,2,3], ylims=(0.8, 110))
	plot!([NaN], [NaN], linewidth=1, linestyle=:dot, label="Slice")
	plot!(ws, step_shrink_cost.(ws, 1.0), label=nothing, linestyle=:dash)
	plot!([NaN], [NaN], linewidth=1, linestyle=:dash, label="Shrink")
	plot!(ws, step_cost_total.(ws), label=nothing, linestyle=:solid) 
	plot!([NaN], [NaN], linewidth=1, linestyle=:solid, label="Total")
	savefig(p, "stepping_costs.png")
	display(p)
	readline()

	println("Doubling")

	p = plot(ws, double_slice_cost.(ws, 1.0), label=nothing, linestyle=:dot) # , yticks= 10.0 .^[-4,-3,-2,-1,0,1,2], legend=:bottomright, ylims=(0.08, 70))
	plot!([NaN], [NaN], linewidth=1, linestyle=:dot, label="Slice")
	plot!(ws, lazydouble_slice_cost.(ws, 1.0), label=nothing, color=:grey, linestyle=:dot)
	plot!(ws, double_shrink_cost.(ws, 1.0), label=nothing, linestyle=:dash)
	plot!([NaN], [NaN], linewidth=1, linestyle=:dash, label="Shrink")
	plot!(ws, double_accept_cost.(ws, 1.0), label=nothing, linestyle=:dashdot)
	plot!([NaN], [NaN], linewidth=1, linestyle=:dashdot, label="Accept")
	plot!(ws, lazydouble_accept_cost.(ws, 1.0), label=nothing, color=:grey, linestyle=:dashdot)
	plot!(ws, double_cost_total.(ws), label=nothing, linestyle=:solid) 
	plot!([NaN], [NaN], linewidth=1, linestyle=:solid, label="Total")
	plot!(ws,lazydouble_cost_total.(ws), label=nothing, linestyle=:solid, color=:grey) 
	plot!([NaN], [NaN], linewidth=1, linestyle=:solid, color=:grey)
	savefig(p, "doubling_costs.png")
	display(p)
	readline()

	println("All")

	p = plot(ws, sum(step_costs_w,dims=2), label=nothing, linestyle=:dash, yscale=:identity, ylims=(0,30))
	plot!(ws, sum(double_costs_w,dims=2), label=nothing, linestyle=:solid, yscale=:identity, ylims=(0,30))
	plot!(ws, sum(lazydouble_costs_w,dims=2), label=nothing, linestyle=:dot, yscale=:identity, ylims=(0,30))
	plot!([NaN], [NaN], linewidth=1, linestyle=:dash, label="Stepping")
	plot!([NaN], [NaN], linewidth=1, linestyle=:solid, label="Cached Doubling")
	plot!([NaN], [NaN], linewidth=1, linestyle=:dot, label="Lazy Cached Doubling")
	savefig(p, "cost_comparison.png")
	display(p)
	readline()

	p = plot(ws, sum(double_costs_w,dims=2)./sum(step_costs_w,dims=2), ylabel="Cost Ratio (vs Stepping Out)", linestyle=:solid, yscale=:identity, legend=:bottomright)
	plot!(ws, sum(lazydouble_costs_w,dims=2)./sum(step_costs_w,dims=2), ylabel="Cost Ratio (vs Stepping Out)", yscale=:identity, linestyle=:dot)
	plot!([NaN], [NaN], linewidth=1, linestyle=:solid, label="Cached Doubling")
	plot!([NaN], [NaN], linewidth=1, linestyle=:dot, label="Lazy Cached Doubling")
	hline!([1.0], linestyle=:dash, color=:grey, label="")
	savefig(p, "cost_ratio.png")
	display(p)
	readline()

end

main()

