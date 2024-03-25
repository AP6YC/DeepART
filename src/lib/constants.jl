"""
    constants.jl

# Description
The constant values for the project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

"""
Definition of the precision used for Flux computations; used for loading data and constructing objects depending on Flux elements.
"""
const FluxFloat = Float32

"""
The default split ration for train/test datasets.
"""
const DEFAULT_P = 0.8

"""
The default shuffle flag for setting up training datasets.
"""
const DEFAULT_SHUFFLE = true

"""
The default number of processes to start in distributed experiments on Windows.
"""
const DEFAULT_N_PROCS_WINDOWS = 11

"""
The default number of processes to start in distributed experiments on Linux.
"""
const DEFAULT_N_PROCS_UNIX = 31

"""
The default plotting dots-per-inch for saving.
"""
const DPI = 600

"""
Plotting linewidth.
"""
const LINEWIDTH = 4.0

"""
Plotting colorscheme.
"""
const COLORSCHEME = :okabe_ito

"""
Plotting fontfamily for all text.
"""
const FONTFAMILY = "Computer Modern"

"""
Heatmap color gradient.
"""
const GRADIENTSCHEME = pubu_9[5:end]

"""
Aspect ratio correction for heatmap
"""
const SQUARE_SIZE = 500.0 .* (1.0, 0.87)  # -8Plots.mm

const percentage_formatter = j -> @sprintf("%0.0f%%", 100*j)

const CONDENSED_LINEWIDTH = 2.5

const N_EB = 8

const DOUBLE_WIDE = 1.0 .* (1200, 400)
