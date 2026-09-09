using Random
using Plots

include("slice_sampling.jl")
include("targets.jl")

function main()
	T = 100_000
	ws = 10.0 .^(-3:0.1:3)
	for i=1:length(ws)
		println("Iteration $i out of $(length(ws))")
		w = ws[i]
		for (target, sample_target) in [(logpdf_mixuniform, sample_mixuniform), (logpdf_mixnormal, sample_mixnormal), (logpdf_mixlaplace, sample_mixlaplace), (logpdf_mixcauchy, sample_mixcauchy)]
		#for (target, sample_target) in [(logpdf_uniform, sample_uniform), (logpdf_normal, sample_normal), (logpdf_laplace, sample_laplace), (logpdf_cauchy, sample_cauchy)]
			for t=1:T
				# draw from the target
				x = sample_target()
				fx = target(x)
				u = log(rand()) + fx
				# do one doubling slice sampling move
				rngstate = copy(Random.default_rng())
				ℓ1, r1, fℓ1, fr1, fcache1, _ = doubling(x, u, w, target)
				copy!(Random.default_rng(), rngstate)
				ℓ2, r2, fℓ2, fr2, fcache2, _ = lazy_doubling(x, u, w, target)
				@assert ℓ1 == ℓ2 && r1 == r2 "ERROR doubling results did not align"

				# need to re-copy the rng state since doubling and lazy_doubling consume different randomness
				rngstate = copy(Random.default_rng())
				xp1, fp1, _, _ = shrinkage(x, u, ℓ1, r1, fℓ1, fr1, w, target, fcache1, :doubling, :default)
				copy!(Random.default_rng(), rngstate)
				xp2, fp2, _, _ = shrinkage(x, u, ℓ1, r1, fℓ1, fr1, w, target, fcache1, :doubling, :original)
				copy!(Random.default_rng(), rngstate)
				xp3, fp3, _, _ = shrinkage(x, u, ℓ2, r2, fℓ2, fr2, w, target, fcache2, :lazydoubling, :default)
				copy!(Random.default_rng(), rngstate)
				xp4, fp4, _, _ = shrinkage(x, u, ℓ2, r2, fℓ2, fr2, w, target, fcache2, :lazydoubling, :original)
				@assert (xp1 == xp2 == xp3 == xp4) "ERROR shrinkage results did not align"
			end
		end
	end
	println("Completed accept check; errors (if any) will be listed above")
end

main()


