# Lazy slice sampling

Implementation of hybrid slice sampling with stepping out and doubling,
and automated near-optimal slice-adaptive tuning methods for each.
This implementation uses caching and lazy evaluation to prevent all unnecessary target evaluations.

This code repository accompanies the preprint

[T. Campbell, "Optimal slice-adaptive tuning of hybrid slice sampling." arXiv:2609.08172.](https://arxiv.org/abs/2609.08172)

Before using this code, open a Julia terminal in this repository, and activate the environment / instantiate it if necessary.
Below is a brief description of all code files. The `minimize_w.jl`, `check_accepts.jl`, and all `plot_*.jl` files are
runnable scripts, while the other files are just function definitions.

- `slice_sampling.jl` contains the main implementation of the methods described in the above preprint.
- `cost_formulae.jl` contains all exact / surrogate cost formulae.
- `targets.jl` contains sampling/log-density functions for the targets in the simulation experiments.
- `minimize_w.jl` is used to find the oracle-optimal setting of the initial window (Theorems 3.6, 3.8, 3.9) .
- `check_accepts.jl` is used to ensure that the lazy cached doubling/accept code returns the same result as the original implementations in Neal, "Slice sampling." Annals of Statistics 31, 2003.
- `plot_*.jl` contains plotting code for all the figures in the preprint.

Two important notes:

1. This code is released for reproducibility purposes and is not production quality.
Some of the `plot_*.jl` scripts in particular may require manual tweaking to produce slightly different 
versions of results, etc.

2. This code follows the implementations in the above preprint, and so minimizes the number of target evaluations, but otherwise 
is not particularly carefully engineered, so major efficiency gains could probably be obtained by just following a few
basic principles [from the Julia docs](https://docs.julialang.org/en/v1/manual/performance-tips/).
