using JLD2, Interpolations
using Random, Distributions

include("slice_sampling.jl")

function lazydouble_cost_estimate(w, λ)
	f(x) = 0 ≤ x ≤ 1 ? 0.0 : -Inf
	T = 1000_000_000
	cslice = 0
	cshrink = 0
	caccept = 0
	σslice = 0
	σshrink = 0
	σaccept = 0
	println("")
	for t=1:T
		if t % 100000 == 0
			print("iteration $t / $T            \r")
		end
		x = rand()
		fx = 0.0
		_, _, cost_slice, cost_shrink, cost_accept = slice_sample(x, ConstantW(w), f, fx, :lazydoubling, :default)
		δslice = cost_slice - cslice 
		δshrink = cost_shrink - cshrink 
		δaccept = cost_accept - caccept
		cslice += δslice/t
		cshrink += δshrink/t
		caccept += δaccept/t
		σslice += δslice*(cost_slice - cslice)
		σshrink += δshrink*(cost_shrink - cshrink)
		σaccept += δaccept*(cost_accept - caccept)
	end
	return cslice, cshrink, caccept, σslice/T^2, σshrink/T^2, σaccept/T^2
end

function create_lazycached_cost()
	if isfile("lazycached.jld2")
		d = load("lazycached.jld2")
		return linear_interpolation(log.(d["ws"]), d["cslices"]), linear_interpolation(log.(d["ws"]), d["cshrinks"]), linear_interpolation(log.(d["ws"]), d["caccepts"])
	end
	ws = sort(unique(vcat(2.2:0.001:2.3, 10.0 .^(-6:0.01:6), 10.0 .^(-15:1:15))))
	f(x) = 0 ≤ x ≤ 1 ? 0.0 : -Inf
	cslices = zero(ws)
	cshrinks = zero(ws)
	caccepts = zero(ws)
	for i=1:length(ws)
		println("iteration $i / $(length(ws))")
		cslices[i], cshrinks[i], caccepts[i], _, _, _ = lazydouble_cost_estimate(ws[i], 1.0)
	end
	jldsave("lazycached.jld2"; ws, cslices, cshrinks, caccepts)
	return linear_interpolation(log.(ws), cslices), linear_interpolation(log.(ws), cshrinks), linear_interpolation(log.(ws), caccepts)
end

step_slice_cost(w,λ) = 2 + λ/w
step_accept_cost(w,λ) = 0
step_shrink_cost(w,λ) = 2.0*(λ/w+1)*log(1+w/λ) - 1.0

function double_slice_cost(w,λ)
	n0 = max(0, ceil(log2(λ/w)))
	c0 = λ/(w*2^n0)
	return 2 + n0 + 2c0
end
function double_accept_cost(w,λ)
	n0 = max(0, ceil(log2(λ/w)))
	c0 = λ/(w*2^n0)
	return c0/3 + max(0, c0/3+n0-1-(1-2.0^(1-n0))/c0 + (1-4.0^(1-n0))/(9c0^2))
end
function double_shrink_cost(w,λ)
	n0 = max(0, ceil(log2(λ/w)))
	c0 = λ/(w*2^n0)
	return 1 + 2*(log(2)*c0 - (1+c0)*log(c0) -1 + 0.5*sum([ (1+c0/2^j)*log(1+c0/2^j) for j=0:50]))
end
lazydouble_slice_cost(w,λ) = lazy_slice_interp(log.(w/λ))
lazydouble_shrink_cost(w,λ) = lazy_shrink_interp(log.(w/λ))
lazydouble_accept_cost(w,λ) = lazy_accept_interp(log.(w/λ))

step_cost_approximation_separable(w,λ) = 3.6 + (6/5)λ/w-log(λ/w)
double_cost_approximation_convex(w,λ) = (2/log(2)+2)*log(1+(2+log(2)/2)*λ/w) - 2*log(λ/w)+1.2
double_cost_approximation_separable(w,λ) = 4 + (λ/w > 0.3 ? 2/log(2)*log(λ/w) - 2/log(2)*log(0.3) : -2log(λ/w) +2log(0.3))
lazydouble_cost_approximation_convex(w,λ) = (5/(6*log(2))+2)*log(1+3*(2+log(2))*λ/w/2) - 2log(3*λ/w/2) + 1.5 
lazydouble_cost_approximation_separable(w,λ) = 4 + (λ/w > (1/3) ? 5/(6*log(2))*log(3λ/w) : -2log(3λ/w))

double_cost_total(w) = double_slice_cost(w,1) + double_accept_cost(w,1) + double_shrink_cost(w,1)
lazydouble_cost_total(w) = lazydouble_slice_cost(w,1) + lazydouble_accept_cost(w,1) + lazydouble_shrink_cost(w,1)
step_cost_total(w) = step_slice_cost(w,1) + step_accept_cost(w,1) + step_shrink_cost(w,1)

lazy_slice_interp, lazy_shrink_interp, lazy_accept_interp = create_lazycached_cost()
