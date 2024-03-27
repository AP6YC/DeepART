"""
    analyze_complex_alt.jl

# Description
This script analyzes a complex single condensed scenario iteration.
This script is updated to use the updated full condensed scenario plot.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "3_single_condensed"

# Saving names
plot_name = "3_single_condensed.png"

# Load name
data_file = DeepART.results_dir(experiment_top, "condensed_complex_data.jld2")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

# Load the data used for generating the condensed scenario plot
perfs, vals, class_labels = DeepART.load_sim_results(data_file, "perfs", "vals", "class_labels")
# class_labels = vcat("", class_labels)

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

# Alternative simplified condensed scenario plot
# p = create_condensed_plot(perfs, class_labels)
p, training_vals, x_training_vals = DeepART.create_complex_condensed_plot_alt(
    perfs, vals, class_labels
)
# DeepART.handle_display(p, pargs)
p
# Save the plot
DeepART.saveplot(
    p,
    "single_condensed",
    [experiment_top,],
    paper=Sys.iswindows(),
    extension=".png",
)
