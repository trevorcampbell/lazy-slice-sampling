using Random
using Plots
using Statistics

include("slice_sampling.jl")
include("targets.jl")

function evaluate_tuner_mcmc(target, sample_target, w_tuner, slice_type, T)
	costs = zeros(T)
	tuners = [deepcopy(w_tuner)]
	# draw from the target to initialize
	x = sample_target()
	fx = target(x)
	for t=1:T
		# do one slice sampling move
		x, fx, cost_slice, cost_shrink, cost_accept = slice_sample(x, w_tuner, target, fx, slice_type)
		costs[t] = cost_slice+cost_shrink+cost_accept
		if ispow2(t)
			tune!(w_tuner, slice_type)
			push!(tuners, deepcopy(w_tuner))
		end
	end
	costs, tuners
end

function compare_tuner_cost(target, sample_target, ws, T, slice_type)
	constant_costs = zeros(length(ws), T)
	oneshot_costs = zeros(length(ws), T)
	gradient_costs = zeros(length(ws), T)
	for i=1:length(ws)
		println("scanning over w: iteration $i out of $(length(ws))")
		w = ws[i]
		println("Constant w")
		constant_cost, _ = evaluate_tuner_mcmc(target, sample_target, ConstantW(w), slice_type, T)
		constant_costs[i, :] = constant_cost
		println("OneShot w")
		oneshot_cost, _ = evaluate_tuner_mcmc(target, sample_target, OneShotW(w), slice_type, T)
		oneshot_costs[i, :] = oneshot_cost
		println("Gradient w")
		gradient_cost, _ = evaluate_tuner_mcmc(target, sample_target, GradientW(w), slice_type, T)
		gradient_costs[i, :] = gradient_cost
	end
	return constant_costs, oneshot_costs, gradient_costs
end

function main()
	T = 100_000
	ws = 10.0 .^(-3:0.05:3)
	default(
	titlefontsize = 18,
    	guidefontsize = 16,
    	tickfontsize = 14,
    	legendfontsize = 14,
    	linewidth=2,
    	label=nothing,
    	xscale=:log10,
    	yscale=:identity, 
    	xlabel="w",
		color=:black,
    	ylabel="Average Cost per Iteration",
 	dpi=200
	)

	for slice_type in [:lazydoubling]
		p1 = plot(legend=nothing)
		if slice_type == :stepping
			#yticks= 10.0 .^[-1,0,1,2,3,4,5]
			#ylims=(2, 200000)
			ylims=(0,50)
			optimal_cost = 4.715
		else
			#yticks= 10.0 .^[-1,0,1,2,3,4,5]
			#ylims=(4, 200)
			ylims=(0,20)
			optimal_cost = 5.410
		end
		i = 1
		for (target, sample_target) in [(logpdf_mixuniform, sample_mixuniform), (logpdf_mixnormal, sample_mixnormal), (logpdf_mixlaplace, sample_mixlaplace), (logpdf_mixcauchy, sample_mixcauchy)]
			println("Target $i")
			constant_costs, oneshot_costs, gradient_costs = compare_tuner_cost(target, sample_target, ws, T, slice_type)
			plot!(p1, ws, mean(constant_costs,dims=2), linestyle=:solid, alpha=i/4, ylims=ylims)#, yticks=yticks)
			plot!(p1, ws, mean(oneshot_costs,dims=2), linestyle=:dash, alpha=i/4)
			plot!(p1, ws, mean(gradient_costs,dims=2), linestyle=:dot, alpha=i/4)
			i += 1
		end
 		hline!(p1, [optimal_cost], color=:grey, linestyle=:dash)

		# Color legend
		plot!(p1, [NaN], [NaN], alpha=1/4, linestyle=:solid, label="Uniform Mix")
		plot!(p1, [NaN], [NaN], alpha=2/4, linestyle=:solid, label="Normal Mix")
		plot!(p1, [NaN], [NaN], alpha=3/4, linestyle=:solid, label="Laplace Mix")
		plot!(p1, [NaN], [NaN], alpha=4/4, linestyle=:solid, label="Cauchy Mix")

		# Style legend
		plot!(p1, [NaN], [NaN], color=:black, linewidth=1, linestyle=:solid, label="Constant")
		plot!(p1, [NaN], [NaN], color=:black, linewidth=1, linestyle=:dash,  label="One Shot")
		plot!(p1, [NaN], [NaN], color=:black, linewidth=1, linestyle=:dot,  label="Gradient")

		# optimal legend
		plot!(p1, [NaN], [NaN], color=:grey, linestyle=:dash, linewidth=1, label="Optimal")

		savefig(p1, "tuning_cost_comparison_multimod_$(string(slice_type)).png")

		display(p1)
		readline()
	end
	
end

main()


