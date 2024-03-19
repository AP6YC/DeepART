"""
    plots.jl

# Description
Plotting functions and utilities.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

function term_accuracy(
    # accs::Vector{Vector{Float}},
    accs::Vector{Any},
    # labels::Vector{String},
    # title::String,
)
    p = lineplot(
        accs,
        title="Accuracy Trend",
        xlabel="Iteration",
        ylabel="Test Accuracy",
    )

    return p
end

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

"""
Creates the confusion matrix as a heatmap using `Plots`.

# Arguments
- `class_labels::Vector{String}`: the string labels for the classes.
- `y::IntegerVector`: the class truth values.
- `y_hat::IntegerVecto`: the class estimates.
"""
function create_confusion_heatmap(
    class_labels::Vector{String},
    y::IntegerVector,
    y_hat::IntegerVector,
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
        dpi=DPI
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
    p,
    filename,
    parts...
)
    # paper_out_dir(args...) = DeepART.paper_results_dir("instart", args...)
    paper_out_dir(args...) = DeepART.paper_results_dir(parts..., args...)
    mkpath(paper_out_dir())
    Plots.savefig(p, paper_out_dir(filename))
end
