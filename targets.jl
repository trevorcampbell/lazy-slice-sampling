using Random
using Distributions
using LogExpFunctions

# univariate targets
logpdf_uniform(x) = (0 ≤ x ≤ 1) ? 0.0 : -Inf
sample_uniform() = rand()
slice_uniform(logu) = (logu ≤ 0.0) ? (0.0, 1.0) : error("u = $u outside of range for uniform")
max_uniform() = 0.0
logpdf_normal(x) = -x^2/2 - 0.5log(2π)
sample_normal() = randn()
slice_normal(logu) = (logu ≤ -0.5log(2π)) ? (-sqrt(-2(0.5log(2π) + logu)), sqrt(-2(0.5log(2π) + logu))) : error("u = $u outside of range for normal") 
max_normal() = -0.5log(2π)
logpdf_laplace(x) = log(0.5) -abs(x)
sample_laplace() = (rand() ≥ 0.5 ? 1 : -1)*log(rand())
slice_laplace(logu) = (logu ≤ log(0.5)) ? (log(2) + logu, -(log(2) + logu)) : error("u = $u outside of range for laplace")
max_laplace() = log(0.5)
logpdf_cauchy(x) = -log(π) - log(1+x^2)
sample_cauchy() = tan(π*(rand()-1/2))
slice_cauchy(logu) = (logu ≤ -log(π)) ? (-sqrt(1/(π*exp(logu))-1), sqrt(1/(π*exp(logu))-1)) : error("u = $u outside of range for Cauchy")
max_cauchy() = -log(π)

logpdf_gamma(x,α) = logpdf(Gamma(α, 1),x)
sample_gamma(α) = rand(Gamma(α,1))
logpdf_student(x,ν) = logpdf(TDist(ν), x)
sample_student(ν) = rand(TDist(ν))

# multivariate targets
logpdf_multiuniform(x) = sum(logpdf_uniform.(x))
sample_multiuniform(d) = [sample_uniform() for i in 1:d]
logpdf_multinormal(x) = sum(logpdf_normal.(x))
sample_multinormal(d) = [sample_normal() for i in 1:d]
logpdf_multilaplace(x) = sum(logpdf_laplace.(x))
sample_multilaplace(d) = [sample_laplace() for i in 1:d]
logpdf_multicauchy(x) = sum(logpdf_cauchy.(x))
sample_multicauchy(d) = [sample_cauchy() for i in 1:d]

# multimodal targets
logpdf_mixuniform(x) = logsumexp([log(0.4)+logpdf_uniform(x+3/2), log(0.2) + logpdf_uniform(x+1/2), log(0.4) + logpdf_uniform(x-1/2)])
sample_mixuniform() = [sample_uniform()-3/2, sample_uniform()-1/2, sample_uniform()+1/2][rand(Categorical([0.4,0.2,0.4]))]
logpdf_mixnormal(x) = logsumexp([log(0.5)+logpdf_normal(x-1), log(0.5)+logpdf_normal(x+1)])
sample_mixnormal() = [sample_normal()+1, sample_normal()-1][rand(Categorical([0.5,0.5]))]
logpdf_mixlaplace(x) = logsumexp([log(0.5)+logpdf_laplace(x-1), log(0.5)+logpdf_laplace(x+1)])
sample_mixlaplace() = [sample_laplace()+1, sample_laplace()-1][rand(Categorical([0.5,0.5]))]
logpdf_mixcauchy(x) = logsumexp([log(0.5)+logpdf_cauchy(x-1), log(0.5)+logpdf_cauchy(x+1)])
sample_mixcauchy() = [sample_cauchy()+1, sample_cauchy()-1][rand(Categorical([0.5,0.5]))]

