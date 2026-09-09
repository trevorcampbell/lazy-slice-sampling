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
		x, fx, cost_slice, cost_shrink, cost_accept = slice_sample(x, w_tuner, target, fx, slice_type, :cached)
		costs[t] = cost_slice+cost_shrink+cost_accept
		if ispow2(t)
			tune!(w_tuner, slice_type)
			push!(tuners, deepcopy(w_tuner))
		end
	end
	costs, tuners
end

function compare_tuner_tuning(target, sample_target, T, slice_type)
	_, oneshot_tuners = evaluate_tuner_mcmc(target, sample_target, OneShotW(1.0), slice_type, T)
	_, gradient_tuners = evaluate_tuner_mcmc(target, sample_target, GradientW(1.0), slice_type, T)
	return oneshot_tuners, gradient_tuners
end

function remove_duplicate_within_sequence(x,y)
	idx = findall([i == 1 || x[i] != x[i-1] || i == length(x) || x[i] != x[i+1] for i in eachindex(x)])
	return x[idx], y[idx]
end

function main()
	T = 10_000
	default(
    titlefontsize = 18,
    guidefontsize = 16,
    tickfontsize = 14,
    legendfontsize = 14,
    linewidth=1,
    label=nothing,
    xscale=:log10,
    yscale=:log10,
    xlabel="w",
    ylabel="u",
 	dpi=200
	)

	for slice_type in [:stepping, :lazydoubling]
		for (target, sample_target, slice_target, max_target) in [(logpdf_uniform, sample_uniform, slice_uniform, max_uniform), (logpdf_normal, sample_normal, slice_normal, max_normal), (logpdf_laplace, sample_laplace, slice_laplace, max_laplace), (logpdf_cauchy, sample_cauchy, slice_cauchy, max_cauchy)]

			us = 10.0 .^(-6:0.001:log10(max_target()))
			logus = log.(us)
			λs = [s[2]-s[1] for s in slice_target.(logus)]
			p = plot(λs, us, color=:black)
			if slice_type == :stepping
				wopts = 1.358λs
			elseif slice_type == :doubling
				wopts = 3.211λs
			elseif slice_type == :lazydoubling
				wopts = 2.277λs
			else
				error("Unknown slice type")
			end
			plot!(wopts, us, color=:black, linestyle=:dash)
			oneshot_tuners, gradient_tuners = compare_tuner_tuning(target, sample_target, T, slice_type)
			i = 1
			for tuner in oneshot_tuners
				ws = get_w.([tuner], logus)
				ws, uplts = remove_duplicate_within_sequence(ws, us)
				plot!(ws, uplts, color=palette(:default)[1], alpha=(i/length(oneshot_tuners))^2)
				i += 1
			end
			i = 1
			for tuner in gradient_tuners
				ws = get_w.([tuner], logus)
				ws, uplts = remove_duplicate_within_sequence(ws, us)
				plot!(ws, uplts, color=palette(:default)[2], alpha=(i/length(gradient_tuners))^2)
				i += 1
			end

			savefig(p, "tuning_tuner_comparison.png")
			display(p)
			readline()
		end

		## Color legend
		#plot!([NaN], [NaN], color=palette(:default)[1], linestyle=:solid, label="Uniform")
		#plot!([NaN], [NaN], color=palette(:default)[2], linestyle=:solid, label="Normal")
		#plot!([NaN], [NaN], color=palette(:default)[3], linestyle=:solid, label="Laplace")
		#plot!([NaN], [NaN], color=palette(:default)[4], linestyle=:solid, label="Cauchy")

		## Style legend
		#plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:solid, label="Constant")
		#plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dash,  label="One Shot")
		#plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dot,  label="Gradient")
	end
	
end

main()


