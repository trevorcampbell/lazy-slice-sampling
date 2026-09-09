using Random, Statistics, Optim

function stepping_out(x, u, w, f, rng)
	V = rand(rng)
	ℓ = x-V*w
	fℓ = f(ℓ)
	r = x + (1-V)*w
	fr = f(r)
	cost = 2
	while fr ≥ u
		r += w
		fr = f(r)
		cost += 1
	end
	while fℓ ≥ u
		ℓ -= w
		fℓ = f(ℓ)
		cost += 1
	end
	return ℓ, r, fℓ, fr, Vector{Float64}(), cost
end

function doubling(x, u, w, f, rng)
	V = rand(rng)
	ℓ = x-V*w
	r = x+(1-V)*w
	fℓ = f(ℓ)
	fr = f(r)
	cost = 2
	fcache = Vector{Float64}()
	while fℓ ≥ u || fr ≥ u
		Z = rand(rng)
		if Z ≤ 0.5
			push!(fcache, fℓ)
			ℓ -= (r-ℓ)
			fℓ = f(ℓ)
		else
			push!(fcache, fr)
			r += (r-ℓ)
			fr = f(r)
		end
		cost += 1
	end
	return ℓ, r, fℓ, fr, fcache, cost
end

function lazy_doubling(x, u, w, f, rng)
	V = rand(rng)
	ℓ = x-V*w
	r = x+(1-V)*w
	fℓ = NaN
	fr = NaN
	cost = 0
	fcache = Vector{Float64}()
	while true
		Z = rand(rng)
		if isnan(fℓ) && isnan(fr)
			if Z ≤ 0.5
				fr = f(r)
				cost += 1
			else
				fℓ = f(ℓ)
				cost += 1
			end
		end
		if !isnan(fℓ) && fℓ < u && isnan(fr)
			fr = f(r)
			cost += 1
		end
		if !isnan(fr) && fr < u && isnan(fℓ)
			fℓ = f(ℓ)
			cost += 1
		end
		if !isnan(fr) && !isnan(fℓ) && fr < u && fℓ < u
			return ℓ, r, fℓ, fr, fcache, cost
		end
		if Z ≤ 0.5
			push!(fcache, fℓ)
			ℓ -= (r-ℓ)
			fℓ = NaN
		else
			push!(fcache, fr)
			r += (r-ℓ)
			fr = NaN
		end
	end
end

function shrinkage(x, u, ℓ, r, fℓ, fr, w, f, fcache, slice_type, accept_type, rng)
	ℓp = ℓ
	rp = r
	cost_shrink = 0
	cost_accept = 0
	while true
		V = rand(rng)
		xp = V*ℓp + (1-V)*rp
		fp = f(xp)
		cost_shrink += 1
		if fp ≥ u
			if slice_type == :stepping
				acc = true
				c = 0
			elseif accept_type == :default && slice_type == :doubling
				acc, c = accept(x, xp, u, ℓ, r, fℓ, fr, w, f, fcache)
			elseif accept_type == :default && slice_type == :lazydoubling
				acc, c = lazy_accept(x, xp, u, ℓ, r, fℓ, fr, w, f, fcache)
			elseif accept_type == :original
				acc, c = accept_original(x, xp, u, ℓ, r, w, f)
			else
				error("Unknown acceptance function type!")
			end
			cost_accept += c
			!acc || return xp, fp, cost_shrink, cost_accept
		end
		if xp < x
			ℓp = xp
		else
			rp = xp
		end	
	end
end

function accept(x, xp, u, ℓ, r, fℓ, fr, w, f, fcache)
	cacheidx = length(fcache)
	cost = 0
	while cacheidx > 0
		# compute the midpoint
		m = (r+ℓ)/2.0	
		# move the left/right boundary as necessary
		# if x remains in the slice prior to the upcoming move, we can still use the cache
		use_cache = ℓ ≤ x ≤ r
		if xp < m
			r = m
			fr = use_cache ? fcache[cacheidx] : f(r)
			cost += !use_cache
		else
			ℓ = m
			fℓ = use_cache ? fcache[cacheidx] : f(ℓ)
			cost += !use_cache
		end
		cacheidx -= 1
			
		# try to skip evals. If we know one of the edges is in the slice, we can just continue
		if fℓ < u && fr < u
			return false, cost
		end
	end
	return true, cost
end

function lazy_accept(x, xp, u, ℓ, r, fℓ, fr, w, f, fcache)
	cacheidx = length(fcache)
	cost = 0
	while cacheidx > 0
		# compute the midpoint
		m = (r+ℓ)/2.0	
		# move the left/right boundary as necessary
		# if x remains in the slice prior to the upcoming move, we can still use the cache
		use_cache = ℓ ≤ x ≤ r
		if xp < m
			r = m
			fr = use_cache ? fcache[cacheidx] : NaN
		else
			ℓ = m
			fℓ = use_cache ? fcache[cacheidx] : NaN
		end
		cacheidx -= 1

		# try to skip evals. If we know one of the edges is in the slice, we can just continue
		if (!isnan(fℓ) && fℓ ≥ u) || (!isnan(fr) && fr ≥ u)
			continue
		end

		# we're forced to evaluate. both fr,fl are unknown or < u.
		# in the event that only one is unknown, the below will always cost 1
		# in the event that both are unknown, it evaluates the edge that didn't move first
		if isnan(fℓ) && isnan(fr)
			if xp < m
				fℓ = f(ℓ)
				cost += 1
			else
				fr = f(r)
				cost += 1
			end
		end
		if !isnan(fℓ) && u > fℓ && isnan(fr)
			fr = f(r)
			cost += 1
		end
		if !isnan(fr) && u > fr && isnan(fℓ)
			fℓ = f(ℓ)
			cost += 1
		end
		if !isnan(fℓ) && !isnan(fr) && u > fℓ && u > fr
			return false, cost
		end
	end
	return true, cost
end

function accept_original(x, xp, u, ℓ, r, w, f)
	D = false
	cost = 0
	while r-ℓ > 1.1w
		m = (r+ℓ)/2.0
		if xp < m
			r = m
		else
			ℓ = m
		end
		if !D && (x < m && xp >= m) || (x >= m && xp < m)
			D = true
		end
		# the next part is equivalent to
		#if D && u > f(ℓ) && u > f(r)
		#	return false, cost
		#end
		# but since we're tracking costs, we need to account for short-circuit evals
		if D 
			cost += 1
			if u > f(ℓ)
				cost += 1
				if u > f(r)
					return false, cost
				end
			end
		end
	end
	return true, cost
end

# slice_type can be stepping, doubling, lazydoubling
# accept_type can be default, original
function slice_sample(x, w_tuner, f, fx=missing, slice_type=:lazydoubling, accept_type=:default, rng=Random.default_rng())
	cost_shrink = 0
	cost_accept = 0
	cost_slice = ismissing(fx) ? 1 : 0
	fx = ismissing(fx) ? f(x) : fx
	u = log(rand(rng)) + fx
	f_tracked = BoundsTracker(f, u, x)
	w = get_w(w_tuner, u)
	if slice_type == :stepping
		ℓ, r, fℓ, fr, fcache, cost_slice = stepping_out(x, u, w, f_tracked,rng)
	elseif slice_type == :doubling
		ℓ, r, fℓ, fr, fcache, cost_slice = doubling(x, u, w, f_tracked,rng)
	elseif slice_type == :lazydoubling
		ℓ, r, fℓ, fr, fcache, cost_slice = lazy_doubling(x, u, w, f_tracked,rng)
	else
		error("Unknown slice type")
	end
	xp, fp, cost_shrink, cost_accept = shrinkage(x, u, ℓ, r, fℓ, fr, w, f_tracked, fcache, slice_type, accept_type,rng)
	ℓbound_ℓ, ℓbound_r, rbound_ℓ, rbound_r = bounds(f_tracked)
	collect!(w_tuner, u, ℓbound_ℓ, ℓbound_r, rbound_ℓ, rbound_r)
	return xp, fp, cost_slice, cost_shrink, cost_accept
end

mutable struct BoundsTracker{F <: Function}
	f::F
	u::Float64
	bℓ::Float64
	ar::Float64
	outside::Vector{Float64}
end

BoundsTracker(f::Function, u::Float64, x::Float64) = BoundsTracker(f, u, x, x, Float64[])

function (t::BoundsTracker)(y)
	fy = t.f(y)
	if fy ≥ t.u
		t.bℓ = min(t.bℓ, y)
		t.ar = max(t.ar, y)
	else
		push!(t.outside, y)
	end
	return fy
end

function bounds(t::BoundsTracker)
	# the minimum/maximum of in_slice set upper/lower bounds on left/right edge
	# the max of out_slice that is still < bℓ is aℓ, and vice versa for br
	aℓ = -Inf
	br = Inf
	for i=1:length(t.outside)
		y = t.outside[i]
		aℓ = y < t.bℓ ? max(aℓ, y) : aℓ 
		br = y > t.ar ? min(br, y) : br
	end
	return aℓ, t.bℓ, t.ar, br
end

abstract type WTuner end

struct ConstantW <: WTuner
	w::Float64
end
collect!(w_tuner::ConstantW, u, ℓbound_ℓ, ℓbound_r, rbound_ℓ, rbound_r) = nothing
tune!(w_tuner::ConstantW, slice_type) = nothing
get_w(w_tuner::ConstantW, u) = w_tuner.w
deepcopy(w_tuner::ConstantW) = ConstantW(w_tuner.w)

mutable struct TunedConstantW <: WTuner
	w::Float64
	ubds::Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}
end
TunedConstantW(w::Float64) = TunedConstantW(w, [])
get_w(w_tuner::TunedConstantW, u) = w_tuner.w
deepcopy(w_tuner::TunedConstantW) = TunedConstantW(w_tuner.w, copy(w_tuner.ubds))
tune!(w_tuner::TunedConstantW, slice_type) = tune!(w_tuner, Val(slice_type))
function tune!(w_tuner::TunedConstantW, ::Val{:doubling})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in w_tuner.ubds]
	empty!(w_tuner.ubds)
	w_tuner.w = (10/3)*quantile(λs, 1/(log(2)+1))
end
function tune!(w_tuner::TunedConstantW, ::Val{:stepping})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in w_tuner.ubds]
	empty!(w_tuner.ubds)
	w_tuner.w = (6/5)*mean(λs)
end
function tune!(w_tuner::TunedConstantW, ::Val{:lazydoubling})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in w_tuner.ubds]
	empty!(w_tuner.ubds)
	w_tuner.w = 3*quantile(λs, 5/(12*log(2)+5))
end

function collect!(w_tuner::WTuner, u, ℓbound_ℓ, ℓbound_r, rbound_ℓ, rbound_r)
	push!(w_tuner.ubds, (u, ℓbound_ℓ, ℓbound_r, rbound_ℓ, rbound_r))
end

function get_w(w_tuner::WTuner, u)
    i = searchsortedfirst(w_tuner.tuned_us, u)
    return w_tuner.tuned_ws[min(i, length(w_tuner.tuned_ws))]
end

deepcopy(w_tuner::V) where {V <: WTuner} = V(copy(w_tuner.ubds), copy(w_tuner.tuned_us), copy(w_tuner.tuned_ws))
(::Type{V})(w::Float64) where {V <: WTuner} = V([], [1.0], [w])

function tune!(w_tuner::WTuner, slice_type)
	ubds = w_tuner.ubds
	t = length(ubds)
	η = 0.95
	β = 0.51
	τ = Int(ceil(t^η))
	# randomly permute the draws
	shuffle!(ubds)
	# power-truncate them to make sure complexity is dominated by sampling
	ubds = ubds[1:τ]
	# sort by u
	sort!(ubds, by= x-> x[1])
	# samples per bin
	k = Int(floor(τ^β))
	# processing
	empty!(w_tuner.tuned_us)
	empty!(w_tuner.tuned_ws)
	for i in 1:k:τ
		# get block of draws
    		block = ubds[i:min(i+k-1, end)]
    		# process block
		push!(w_tuner.tuned_us, block[end][1])
		push!(w_tuner.tuned_ws, tune_inner(block, w_tuner, Val(slice_type)))
	end	
	empty!(w_tuner.ubds)
end

struct OneShotW <: WTuner
	ubds::Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}
	tuned_us::Vector{Float64}
	tuned_ws::Vector{Float64}
end

function tune_inner(block, ::OneShotW, ::Val{:doubling})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in block]
	(10/3)*quantile(λs, 1/(log(2)+1))
end

function tune_inner(block, ::OneShotW, ::Val{:lazydoubling})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in block]
	3*quantile(λs, 5/(12*log(2)+5))
end

function tune_inner(block, ::OneShotW, ::Val{:stepping})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in block]
	(6/5)*mean(λs)
end

struct GradientW <: WTuner
	ubds::Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}
	tuned_us::Vector{Float64}
	tuned_ws::Vector{Float64}
end 

fdouble(x) = (2/log(2)+2)*log(1+(2+log(2)/2)*x) - 2*log(x)
function tune_inner(block, w_tuner::GradientW, ::Val{:doubling})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in block]
	w0 = (10/3)*quantile(λs, 1/(log(2)+1))
	result = optimize(x-> sum(fdouble.(λs .* exp.(-first(x)))), [log(w0)])
	return exp(first(Optim.minimizer(result)))
end

flazy(x) = (5/(6*log(2))+2)*log(1+(3/2)*(2+log(2))*x) - 2*log((3/2)*x)
function tune_inner(block, w_tuner::GradientW, ::Val{:lazydoubling})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in block]
	w0 = 3*quantile(λs, 5/(12*log(2)+5))
	result = optimize(x-> sum(flazy.(λs .* exp.(-first(x)))), [log(w0)])
	return exp(first(Optim.minimizer(result)))
end

fstep(x) = 1+x+2*(1+x)*log(1+1/x)
function tune_inner(block, w_tuner::GradientW, ::Val{:stepping})
	λs = [(b[5]+b[4])/2.0 - (b[3]+b[2])/2.0 for b in block]
	w0 = (6/5)*mean(λs)
	result = optimize(x-> sum(fstep.(λs .* exp.(-first(x)))), [log(w0)])
	return exp(first(Optim.minimizer(result)))
end




