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