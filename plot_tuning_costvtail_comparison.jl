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
		print("Iteration $t/$T f=$fx                 \r")
		# do one slice sampling move
		x, fx, cost_slice, cost_shrink, cost_accept = slice_sample(x, w_tuner, target, fx, slice_type)
		costs[t] = cost_slice+cost_shrink+cost_accept
		if ispow2(t)
			tune!(w_tuner, slice_type)
			push!(tuners, deepcopy(w_tuner))
		end
	end
	println("")
	costs, tuners
end

function compare_tuner_cost(β, w, T, slice_type)
	target(x) = logpdf_gamma(x,β) 
	#target(x) = logpdf_student(x, β)
	sample_target() = sample_gamma(β)
	#sample_target() = sample_student(β)
	println("Constant w")
	constant_cost, _ = evaluate_tuner_mcmc(target, sample_target, TunedConstantW(w), slice_type, T)
	println("OneShot w")
	oneshot_cost, _ = evaluate_tuner_mcmc(target, sample_target, OneShotW(w), slice_type, T)
	println("Gradient w")
	gradient_cost, _ = evaluate_tuner_mcmc(target, sample_target, GradientW(w), slice_type, T)
	return mean(constant_cost), mean(oneshot_cost), mean(gradient_cost)
end

function main()
	T = 1000_000
	w = 1.0
	# gamma shapes / student t dofs
	βs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
	ylims = (0, 45)
	default(
	titlefontsize = 18,
    	guidefontsize = 16,
    	tickfontsize = 14,
    	legendfontsize = 14,
    	linewidth=2,
    	label=nothing,
    	xscale=:identity,
    	yscale=:identity, 
    	xlabel="α",
		color=:black,
    	ylabel="Average Cost per Iteration",
 		dpi=200
	)

	for slice_type in [:lazydoubling]
		p = plot()
		if slice_type == :stepping
			optimal_cost = 4.715
		else
			optimal_cost = 5.410
		end
		constant_costs = zero(βs)
		oneshot_costs = zero(βs)
		gradient_costs = zero(βs)
		for i=1:length(βs)
			β = βs[i]
			constant_costs[i], oneshot_costs[i], gradient_costs[i] = compare_tuner_cost(β, w, T, slice_type)
			
		end
		plot!(p, βs, constant_costs, linestyle=:solid, ylims=ylims)#, yticks=yticks)
		plot!(p, βs, oneshot_costs, linestyle=:dash)
		plot!(p, βs, gradient_costs, linestyle=:dot)
 		hline!(p, [optimal_cost], color=:grey, linestyle=:dash)
		
		# Style legend
		plot!(p, [NaN], [NaN], color=:black, linewidth=1, linestyle=:solid, label="Constant (Tuned)")
		plot!(p, [NaN], [NaN], color=:black, linewidth=1, linestyle=:dash,  label="One Shot")
		plot!(p, [NaN], [NaN], color=:black, linewidth=1, linestyle=:dot,  label="Gradient")

		# optimal legend
		plot!(p, [NaN], [NaN], color=:grey, linestyle=:dash, linewidth=1, label="Optimal")

		savefig(p, "tuning_costvtail_$(string(slice_type)).png")

		display(p)
		readline()
	end
	
end

main()


