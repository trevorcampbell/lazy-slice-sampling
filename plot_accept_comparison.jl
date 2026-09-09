using Random
using Plots
using Statistics

include("slice_sampling.jl")
include("targets.jl")

function evaluate_tuner_iid(target, sample_target, slice_type, accept_type, w, T)
	accept_costs = zeros(T)
	slice_costs = zeros(T)
	for t=1:T
		# draw from the target
		x = sample_target()
		fx = target(x)
		# do one slice sampling move
		_, _, slice_costs[t], _, accept_costs[t] = slice_sample(x, ConstantW(w), target, fx, slice_type, accept_type)
	end
	slice_costs, accept_costs
end

function compare_accept_costs(target, sample_target, ws, T)
	mean_accept_cached = zeros(length(ws))
	mean_slice_cached = zeros(length(ws))
	mean_accept_original = zeros(length(ws))
	mean_slice_original = zeros(length(ws))
	mean_accept_lazy = zeros(length(ws))
	mean_slice_lazy = zeros(length(ws))
	t_cached, t_original, t_lazy = 0.0, 0.0, 0.0
	for i=1:length(ws)
		println("Iteration $i out of $(length(ws))")
		w = ws[i]
		t0 = time_ns()
		slice_costs, accept_costs = evaluate_tuner_iid(target, sample_target, :doubling, :default, w, T)
		t_cached += time_ns()-t0
		mean_accept_cached[i] = mean(accept_costs)
		mean_slice_cached[i] = mean(slice_costs)
		t0 = time_ns()
		slice_costs, accept_costs = evaluate_tuner_iid(target, sample_target, :doubling, :original, w, T)
		t_original += time_ns() - t0
		mean_accept_original[i] = mean(accept_costs)
		mean_slice_original[i] = mean(slice_costs)
		t0 = time_ns()
		slice_costs, accept_costs = evaluate_tuner_iid(target, sample_target, :lazydoubling, :default, w, T)
		t_lazy += time_ns() - t0
		mean_accept_lazy[i] = mean(accept_costs)
		mean_slice_lazy[i] = mean(slice_costs)
	end
	println("t_cached $(Float64(t_cached)/1e9) t_original $(Float64(t_original)/1e9) t_lazy $(Float64(t_lazy)/1e9)")
	mean_accept_cached, mean_slice_cached, mean_accept_original, mean_slice_original, mean_accept_lazy, mean_slice_lazy
end

function main()
	T = 1000_000
	ws = 10.0 .^(-3:0.1:3)
	default(
    titlefontsize = 18,
    guidefontsize = 16,
    tickfontsize = 14,
    legendfontsize = 14,
    linewidth=2,
    label=nothing,
    xscale=:log10,
    yscale=:identity,
    color=:black,
    xlabel="w/λ",
 	dpi=200
	)

	mean_accept_cached, mean_slice_cached, mean_accept_original, mean_slice_original, mean_accept_lazy, mean_slice_lazy = compare_accept_costs(logpdf_uniform, sample_uniform, ws, T)
	# accept plot
	p = plot(ylabel="Average Accept Evals")
	plot!(ws, mean_accept_lazy, linestyle=:solid)
	plot!(ws, mean_accept_cached, linestyle=:dash)
	plot!(ws, mean_accept_original, linestyle=:dot)

	# Style legend
	plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dot,  label="Original")
	plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dash,  label="Caching")
	plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:solid,  label="Lazy Caching")

	savefig(p, "accept_comparison.png")

	display(p)
	readline()

	p = plot(ylabel="Average Doubling Evals")
	plot!(ws, mean_slice_lazy, linestyle=:solid)
	plot!(ws, mean_slice_original, linestyle=:dot)

	# Style legend
	plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:dot,  label="Original & Caching")
	plot!([NaN], [NaN], color=:black, linewidth=1, linestyle=:solid,  label="Lazy Caching")

	savefig(p, "slice_comparison.png")

	display(p)
	readline()
end

main()


