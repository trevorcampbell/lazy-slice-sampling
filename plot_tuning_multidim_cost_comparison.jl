using Random
using Plots
using Statistics

include("slice_sampling.jl")
include("targets.jl")

function evaluate_tuner_mcmc(target, sample_target, d, w_tuner, slice_type, T)
	costs = zeros(T)
	tuners = [deepcopy(w_tuner)]
	# draw from the target to initialize
	x = sample_target(d)
	fx = target(x)
	for t=1:T
		# do one slice sampling move
		p = sample_multinormal(d)
		p /= sqrt(sum(p.^2))
		slicetarget(y) = target(x .+ p*y)
		y, fx, cost_slice, cost_shrink, cost_accept = slice_sample(0.0, w_tuner, slicetarget, fx, slice_type)
		x = x .+ p*y
		costs[t] = cost_slice+cost_shrink+cost_accept
		if ispow2(t)
			tune!(w_tuner, slice_type)
			push!(tuners, deepcopy(w_tuner))
		end
	end
	costs, tuners
end

function compare_tuner_cost(target, sample_target, d, ws, T, slice_type)
	mean_constant_cost = zeros(length(ws))
	mean_oneshot_cost = zeros(length(ws))
	mean_gradient_cost = zeros(length(ws))
	for i=1:length(ws)
		println("scanning over w: iteration $i out of $(length(ws))")
		w = ws[i]
		println("Constant w")
		constant_costs, _ = evaluate_tuner_mcmc(target, sample_target, d, ConstantW(w), slice_type, T)
		mean_constant_cost[i] = mean(constant_costs)
		println("OneShot w")
		oneshot_costs, _ = evaluate_tuner_mcmc(target, sample_target, d, OneShotW(w), slice_type, T)
		mean_oneshot_cost[i] = mean(oneshot_costs)
		println("Gradient w")
		gradient_costs, _ = evaluate_tuner_mcmc(target, sample_target, d, GradientW(w), slice_type, T)
		mean_gradient_cost[i] = mean(gradient_costs)
	end
	return mean_constant_cost, mean_oneshot_cost, mean_gradient_cost
end

function main()
	T = 100_000
	ws = 10.0 .^(-3:0.05:3)
	d = 256
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

	for slice_type in [:stepping]
		p = plot()
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
		for (target, sample_target) in [(logpdf_multiuniform, sample_multiuniform), (logpdf_multinormal, sample_multinormal), (logpdf_multilaplace, sample_multilaplace), (logpdf_multicauchy, sample_multicauchy)]
			println("Target $i")
			mean_constant_cost, mean_oneshot_cost, mean_gradient_cost = compare_tuner_cost(target, sample_target, d, ws, T, slice_type)
			plot!(ws, mean_constant_cost, linestyle=:solid, alpha=i/4, ylims=ylims)#, yticks=yticks)
			plot!(ws, mean_oneshot_cost, linestyle=:dash, alpha=i/4)
			plot!(ws, mean_gradient_cost, linestyle=:dot, alpha=i/4)
			i += 1
		end
 		hline!([optimal_cost], color=:grey, linestyle=:dash)

		# Color legend
		plot!([NaN], [NaN], alpha=1/4, linestyle=:solid, label="Uniform")
		plot!([NaN], [NaN], alpha=2/4, linestyle=:solid, label="Normal")
		plot!([NaN], [NaN], alpha=3/4, linestyle=:solid, label="Laplace")
		plot!([NaN], [NaN], alpha=4/4, linestyle=:solid, label="Cauchy")

		# Style legend
		plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:solid, label="Constant")
		plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dash,  label="One Shot")
		plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dot,  label="Gradient")

		# optimal legend
		plot!([NaN], [NaN], color=:grey, linestyle=:dash, linewidth=1, label="Optimal")

		savefig(p, "tuning_cost_comparison_multidim_$(string(slice_type)).png")

		display(p)
		readline()
	end
	
end

main()


