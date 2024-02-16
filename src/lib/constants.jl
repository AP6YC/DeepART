"""
    constants.jl

# Description
The constant values for the project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

"""
The default split ration for train/test datasets.
"""
const DEFAULT_P = 0.7

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
