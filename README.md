[![deepart-header](https://github.com/AP6YC/FileStorage/blob/main/DeepART/header.png?raw=true)][docs-url]

A repository containing implementations and experiments for the upcoming paper _Deep Adaptive Resonance Theory_.

| **Documentation** | **Testing Status** | **Zenodo DOI** |
|:-----------------:|:------------------:|:--------------:|
| [![Docs][docs-img]][docs-url] | [![CI Status][ci-img]][ci-url] | [![DOI][zenodo-img]][zenodo-url] |

[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.10896042.svg
[zenodo-url]: https://zenodo.org/doi/10.5281/zenodo.10896042

[ci-img]: https://github.com/AP6YC/DeepART/workflows/CI/badge.svg
[ci-url]: https://github.com/AP6YC/DeepART/actions?query=workflow%3ACI

[docs-img]: https://img.shields.io/badge/docs-blue.svg
[docs-url]: https://AP6YC.github.io/DeepART/

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Basic Usage](#basic-usage)
- [Attribution](#attribution)
  - [Authors](#authors)
  - [Datasets](#datasets)
  - [Assets](#assets)
  - [Quotes](#quotes)

[julia-lang]: https://julialang.org/
[julia-docs]: https://docs.julialang.org/en/v1/
[drwatson-docs]: https://juliadynamics.github.io/DrWatson.jl/dev/

## Basic Usage

For detailed usage, please read the [documentation][docs-url].

The `DeepART` project is a [`Julia`][julia-lang] project, so its use follows typical [Julia][julia-docs] usage.
The `DeepART` project also utilizes [`DrWatson.jl`][drwatson-docs] for organizing and running simulations.

The library code for the project is contained in `src/`, while all experiments are enumerated in `scripts`.
Each folder therein contains a simple README for the order of running experiments.

For example, after installing Julia on your system, you instantiate this project with

```julia
using Pkg; Pkg.activate(); Pkg.instantiate()
```

and run an experiment interactively with

```julia
include("scripts/1_baselines/single/conv.jl")
```

or through the terminal with

```shell
julia --project="." "scripts/1_baselines/single/conv.jl"
```

## Attribution

### Authors

- Sasha Petrenko - <sap625@mst.edu> - [@AP6YC](https://github.com/AP6YC)

### Datasets

- [Indoor Scene Recognition](https://web.mit.edu/torralba/www/indoor.html)
  - [Direct Link](http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar)

### Assets

- [Deep-learning icons created by Freepik - Flaticon](https://www.flaticon.com/free-icons/deep-learning) ([deep-learning_2080961](https://www.flaticon.com/free-icon/deep-learning_2080961))
- [Unlearned Font](https://www.1001fonts.com/unlearned-font.html)

### Quotes

> To achieve great things, two things are needed: a plan and not quite enough time
>
> --<cite> Leonard Bernstein </cite>
