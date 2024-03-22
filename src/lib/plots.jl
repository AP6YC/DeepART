"""
    plots.jl

# Description
Plotting functions and utilities.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

"""
Terminal plot function for a simple vector of accuracies.
"""
function term_accuracy(
    # accs::Vector{Vector{Float}},
    accs::Vector{T},
    # labels::Vector{String},
    # title::String,
) where T <: Real
    p = lineplot(
        accs,
        title="Accuracy Trend",
        xlabel="Iteration",
        ylabel="Test Accuracy",
    )

    return p
end

"""
Terminal barplot function for category predictions.
"""
function term_preds(
    y_hats::Vector{Int};
    title::String="Prediction Counts"
)
    names = collect(1:10)
    bars = Vector{Int}()
    for ix in eachindex(names)
        push!(bars, sum(y_hats .== ix))
    end
    p = UnicodePlots.barplot(
        string.(names),
        bars,
        title=title,
    )
    return p
end

# function plot_task_accuracy(
#     accs::Vector{Vector{Float64}},
#     labels::Vector{String},
#     title::String,
#     filename::String,
#     save::Bool,
#     show::Bool,
# )
#     # Create the plot
#     p = plot(
#         accs,
#         label=labels,
#         title=title,
#         xlabel="Task",
#         ylabel="Accuracy",
#         legend=:topleft,
#         lw=2,
#         size=(800, 600),
#     )

#     # Save and show
#     save && savefig(p, filename)
#     show && display(p)
# end


"""
Wrapper method for getting the raw confusion matrix.

# Arguments
- `y::IntegerVector`: the target values.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Integer)
    return confusmat(n_classes, y, y_hat)
end

"""
Get the normalized confusion matrix.

# Arguments
- `y::IntegerVector`: the target values.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_normalized_confusion(y::IntegerVector, y_hat::IntegerVector, n_classes::Integer)
    cm = get_confusion(y, y_hat, n_classes)
    total = sum(cm, dims=2)
    norm_cm = cm./total
    return norm_cm
end

function plot_confusion_matrix(
    y::IntegerVector,
    y_hat::IntegerVector,
    class_labels::Vector{String},
    filename::String,
    dir_parts::Vector{String},
    # paper::Bool=false,
    # save::Bool=true;
    # show::Bool=true;
    kwargs...
)
    # # Number of classes from the class labels
    # n_classes = length(class_labels)
    # # Normalized confusion
    # norm_cm = get_normalized_confusion(y, y_hat, n_classes)

    # Generate the GUI heatmap
    p_gui = create_confusion_heatmap(
        class_labels,
        y,
        y_hat;
        kwargs...
    )

    # Try to display
    display(p_gui)

    # Save the GUI heatmap
    saveplot(
        p_gui,
        filename,
        dir_parts,;
        paper=true,
        extension=".png"
    )

    # Generate the terminal heatmap
    p_term = create_unicode_confusion_heatmap(
        class_labels,
        y,
        y_hat;
        kwargs...
    )

    # Try to display
    display(p_term)

    return
end

"""
Makes and returns a unicode confusion heatmap for terminal viewing.
"""
function create_unicode_confusion_heatmap(
    class_labels::Vector{String},
    y::IntegerVector,
    y_hat::IntegerVector;
    kwargs...
)
    # Number of classes from the class labels
    n_classes = length(class_labels)
    # Normalized confusion
    norm_cm = get_normalized_confusion(y, y_hat, n_classes)

    # @info kwargs
    # @info kwargs...
    p = UnicodePlots.heatmap(
        norm_cm,
        title="Normalized Confusion Matrix",
        xlabel="Predicted",
        ylabel="Truth";
        kwargs...
        # zlabel="asdf"
        # color=:heat,
    )
    return p
end

"""
Creates the confusion matrix as a heatmap using `Plots`.

# Arguments
- `class_labels::Vector{String}`: the string labels for the classes.
- `y::IntegerVector`: the class truth values.
- `y_hat::IntegerVector`: the class estimates.
"""
function create_confusion_heatmap(
    class_labels::Vector{String},
    y::IntegerVector,
    y_hat::IntegerVector;
    kwargs...
)
    # Number of classes from the class labels
    n_classes = length(class_labels)
    # Normalized confusion
    norm_cm = get_normalized_confusion(y, y_hat, n_classes)

    # Transpose reflect
    # plot_cm = reverse(norm_cm', dims=1)
    # plot_cm = reverse(norm_cm, dims=1)
    plot_cm = norm_cm
    # Convert to percentages
    plot_cm *= 100.0
    # Transpose the y labels
    x_labels = class_labels
    # y_labels = reverse(class_labels)
    y_labels = class_labels

    # Create the heatmap
    h = Plots.heatmap(
        x_labels,
        y_labels,
        plot_cm,
        fill_z = norm_cm,
        aspect_ratio=:equal,
        color = cgrad(GRADIENTSCHEME),
        clims = (0, 100),
        fontfamily=FONTFAMILY,
        annotationfontfamily=FONTFAMILY,
        size=SQUARE_SIZE,
        dpi=DPI;
        kwargs...
    )

    # Create the annotations
    fontsize = 10
    nrow, ncol = size(norm_cm)
    ann = [
        (
            i-0.5,
            j-0.5,
            text(
                round(plot_cm[j,i], digits=2),
                fontsize,
                FONTFAMILY,
                :white,
                :center,
            )
        )
        for i in 1:nrow for j in 1:ncol
    ]

    # Add the cell annotations
    Plots.annotate!(
        ann,
        linecolor=:white,
        # linecolor=:black,
        fontfamily=FONTFAMILY,
    )

    plot!(
        bottom_margin = -9Plots.mm,
    )

    # Label truth and predicted axes
    Plots.xlabel!("Predicted")
    Plots.ylabel!("Truth")

    # Return the plot handle for display or saving
    return h
end

"""
Wrapper for saving results plots.
"""
function saveplot(
    # p::Plots.Plot,    # Apparently UnicodePlots aren't Plots.Plot
    p,
    filename::AbstractString,
    parts::Vector{String};
    paper::Bool=false,
    extension::AbstractString="png",
)
    # If saving to the paper directly
    if paper
        # paper_out_dir(args...) = DeepART.paper_results_dir("instart", args...)
        paper_out_dir(args...) = DeepART.paper_results_dir(parts..., args...)
        mkpath(paper_out_dir())
        Plots.savefig(p, paper_out_dir(filename * extension))
    end

    # Save locally too
    results_out_dir(args...) = DeepART.results_dir(parts..., args...)
    mkpath(results_out_dir())
    Plots.savefig(p, results_out_dir(filename * extension))

    return
end


# """
# Create and return an alternate complex condensed scenario plot.
# """
# function create_complex_condensed_plot_alt(
#     perfs,
#     vals,
#     class_labels,
#     percentages::Bool=true
# )
#     # Reshape the labels string vector for plotting
#     local_labels = reshape(class_labels, 1, length(class_labels))
#     # Determine if plotting percentages or [0, 1]
#     y_formatter = percentages ? percentage_formatter : :auto
#     # Set all the linewidths
#     linewidths = CONDENSED_LINEWIDTH
#     # Infer the number of classes
#     n_classes = length(class_labels)
#     # Number of experience block sample points
#     n_eb = N_EB
#     # Initialize cutoff x-locations (EB-LB boundaries)
#     cutoffs = []
#     # First EB
#     push!(cutoffs, n_eb)

#     # Old plot data
#     for i = 1:n_classes
#         # Append the training length to the cutoff
#         push!(cutoffs, cutoffs[end] + size(vals[i], 2))
#         # Append the number of experience block "samples"
#         push!(cutoffs, cutoffs[end] + n_eb)
#     end

#     # Just current training data
#     training_vals = []
#     x_training_vals = []
#     tick_locations = []
#     start_point = cutoffs[1]
#     for i = 1:n_classes
#         # Fencepost evaluation values
#         local_vals = vals[i][i, :]
#         push!(local_vals, vals[i][i, end])
#         n_local_vals = length(local_vals)
#         # Add the tick locations as midway along training
#         push!(tick_locations, start_point + floor(Int, n_local_vals/2))
#         # Add the local training vals
#         push!(training_vals, local_vals)
#         # Add the start and stop points of the training vals
#         push!(x_training_vals, collect(start_point:start_point + n_local_vals - 1))
#         # Reset the start point
#         start_point += n_local_vals + n_eb - 1
#     end

#     # Get evaluation lines locations
#     fcut = vcat(0, cutoffs)
#     eval_points = [mean([fcut[i-1], fcut[i]]) for i = 2:2:length(fcut)]

#     # Local colors
#     local_palette = palette(COLORSCHEME)

#     # New training plotlines
#     p = plot(
#         x_training_vals,
#         training_vals,
#         linestyle=:solid,
#         linewidth=linewidths,
#         labels=local_labels,
#         color_palette=local_palette,
#     )

#     # Vertical lines
#     vline!(
#         p,
#         fcut,
#         linewidth=linewidths,
#         linestyle=:solid,
#         fillalpha=0.1,
#         color=:gray25,
#         label="",
#     )

#     # The biggest headache in the world
#     local_colors = [local_palette[1]; local_palette[collect(2:n_classes+1)]]
#     # Eval lines
#     plot!(
#         p,
#         eval_points,
#         markershape=:circle,
#         markersize=3,
#         hcat(perfs...),
#         color_palette=local_colors,
#         linewidth=linewidths,
#         linestyle=:dot,
#         # linestyle=:dash,
#         labels=""
#     )

#     # Vertical spans (gray boxes)
#     vspan!(
#         p,
#         fcut,           # Full cutoff locations, including 0
#         color=:gray25,  # 25% gray from Colors.jl
#         fillalpha=0.1,  # Opacity
#         label="",       # Keeps the spans out of the legend
#     )

#     # Format the plot
#     plot!(
#         size=DOUBLE_WIDE,
#         yformatter=y_formatter,
#         fontfamily=FONTFAMILY,
#         legend=:outerright,
#         legendfontsize=25,
#         thickness_scaling=1,
#         dpi=DPI,
#         xticks=(tick_locations, class_labels),
#         left_margin = 10Plots.mm,
#     )

#     # xlabel!("Training Class")
#     ylabel!("Testing Accuracy")
#     # xticks!(collect(1:length(local_labels)), local_labels)

#     return p, training_vals, x_training_vals
# end
