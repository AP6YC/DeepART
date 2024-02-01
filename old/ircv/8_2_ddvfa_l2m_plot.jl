using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Plots             # All plots
using PlotThemes        # Themes for the plots
using DelimitedFiles    # Loading .csv files

# Select the filename to plot
filename = "test_accuracies.csv"
result_name = "l2m_cont.png"

# Plotting style
default(show = true)
# pyplot()
# theme(:dark)

# Point to the l2m accuracies data
data_dir(args...) = projectdir("work/results/flir_l2m", args...)
# result_file = projectdir("work/results/flir_l2m")
result_file = data_dir(result_name)

# Load the data
test_accuracies = readdlm(data_dir(filename))

# Get the number of samples for the x-axis
n_samples = length(test_accuracies)

# Plot the data
p = plot(dpi=300, leg=false)
plot!(
    p,
    1:n_samples,
    test_accuracies,
    linewidth=3
)

# Decorate the plot, save, and display
xlabel!("Training Sample")
ylabel!("Testing Accuracy")
savefig(p, result_file)
display(p)
