include("cost_formulae.jl")

function golden_section_search(f, a, b; tol=1e-8, maxiter=1000)
    ϕ = (1 + sqrt(5)) / 2
    invϕ = 1 / ϕ

    # Interior points
    c = b - (b - a) * invϕ
    d = a + (b - a) * invϕ

    fc = f(c)
    fd = f(d)

    iter = 0
    while (b - a) > tol && iter < maxiter
        if fc < fd
            b = d
            d = c
            fd = fc

            c = b - (b - a) * invϕ
            fc = f(c)
        else
            a = c
            c = d
            fc = fd

            d = a + (b - a) * invϕ
            fd = f(d)
        end

        iter += 1
    end

    xmin = (a + b) / 2
    return xmin, f(xmin)
end


function stochastic_golden_section_search(f, a, b; tol=1e-5, maxiter=1000)
    ϕ = (1 + sqrt(5)) / 2
    invϕ = 1 / ϕ

    # Interior points
    c = b - (b - a) * invϕ
    d = a + (b - a) * invϕ

    fc, σc = f(c)
    fd, σd = f(d)

    iter = 0
    println("stochastic golden section starting")
    while (b - a) > tol && iter < maxiter
    	println("Iter $iter: bounds: ($a, $b), fc: $((fc, sqrt(σc))), fd: $((fd,sqrt(σd)))")

    	while (fc ≥ fd && fc - 2*sqrt(σc) ≤ fd + 2*sqrt(σd)) || (fd ≥ fc && fc + 2*sqrt(σc) > fd - 2*sqrt(σd))
    		println("Improving estimates")
    		# do variance-minimizing combination of new and old estimate
			if σc > σd
    				fcn, σcn = f(c)
    				αc = σcn/(σc+σcn)
    				fc = (αc*fc + (1-αc)*fcn)
    				σc = αc^2*σc + (1-αc)^2*σcn
			else
    				fdn, σdn = f(d)
					αd = σdn/(σd+σdn)
    				fd = (αd*fd + (1-αd)*fdn)
    				σd = αd^2*σd + (1-αd)^2*σdn
			end
    		println("bounds: ($a, $b), fc: $((fc, 2*sqrt(σc))), fd: $((fd,2*sqrt(σd)))")
    	end
    	println("Estimates good enough, continuing")

        if fc < fd
            b = d
            d = c
            fd = fc
            σd = σc

            c = b - (b - a) * invϕ
            fc, σc = f(c)
        else
            a = c
            c = d
            fc = fd
            σc = σd

            d = a + (b - a) * invϕ
            fd, σd = f(d)
        end

        iter += 1
    end

    xmin = (a + b) / 2
    return xmin, f(xmin)
end

function lazycost(w)
	a, b, c, σa, σb, σc = lazydouble_cost_estimate(w, 1.0)
	return a+b+c, σa+σb+σc 
end

function main()
	wmin, fmin = golden_section_search(step_cost_total, 0.0001, 10.0)
	println("stepping: min wratio $wmin λratio $(1/wmin) fmin $fmin")
	wmin, fmin = golden_section_search(double_cost_total, 0.0001, 10.0)
	println("doubling: min wratio $wmin λratio $(1/wmin) fmin $fmin")
	wmin, fmin = stochastic_golden_section_search(lazycost, .0001, 10.0)
	println("lazy doubling: min wratio $wmin λratio $(1/wmin) fmin $fmin")
end

main()
