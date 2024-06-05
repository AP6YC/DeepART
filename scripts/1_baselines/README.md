# `1_baselines`

This directory contains the baseline experiments for the `DeepART` project, which include the parallel simulations of each method and dataset to calculate performance statistics for the paper.

## Scripts

- `dist.jl`: the main distributed experiment running simple single- and multi-task experiments to get statistics of final performance values for each dataset and method tested.
- `singles.jl`: a set of example experiments of the kind that are running in parallel in `dist.jl`.
- `sfam.jl`: just simple FuzzyARTMAP run on a single dataset as a sanity check.
- `analyze_dist.jl`: a script taking the distributed results of `dist.jl` and generating the tables and figures for the paper.
