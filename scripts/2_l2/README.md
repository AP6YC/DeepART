# `1_baselines`

This folder contains the baseline experiments for the paper, running external modules on the target datasets and learning scenarios.

## Files

- `sfam_l2.jl`: a single example of each of the training/testing experiments that happens in parallel in other scripts.
- `gen_scenarios.jl`: generates all of the L2 scenario files for use during condensed learning scenarios for the l2logger.
- `l2_job.jl`: the main L2 job running each **method** on the datasets and permutations specified in `gen_scenarios.jl`.
This is currently done in serial because the Julia-Python interop is not thread safe.
- `dist_metrics_par.jl`: runs the `l2metrics` terminal commands in parallel to compute all metrics for each generated log.
- `table.jl`: aggregates the l2metrics into a large table.
- `stats.jl`: computes the statistics of the entries of the generated table.
