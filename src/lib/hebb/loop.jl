"""
    loop.jl

# Description
Utilities for training and testing loops.
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

generate_showvalues(val) = () -> [(:val, val)]

generate_showvalues_n_cat(val, n_cat) = () -> [(:val, val), (:n_cat, n_cat)]

function update_view_progress!(
    # ix_iter,
    # interval_vals,
    # ix_vals,
    # vals,
    p,
    loop_dict,
    model,
    data,
)
    if loop_dict["ix_iter"] % loop_dict["interval_vals"] == 0
        loop_dict["vals"][loop_dict["ix_vals"]] = test(model, data)
        loop_dict["ix_vals"] += 1
    end

    # Update progress bar
    report_value = if loop_dict["ix_vals"] > 1
        loop_dict["vals"][loop_dict["ix_vals"] - 1]
    else
        0.0
    end

    loop_dict["ix_iter"] += 1

    # if haskey(model.opts, "blocks")
    if model isa Hebb.BlockNet
        if model.layers[end] isa Hebb.ARTBlock
            next!(p; showvalues=generate_showvalues_n_cat(
                report_value,
                model.layers[end].model.n_categories,
            ))
        else
            next!(p; showvalues=generate_showvalues(report_value))
        end
    else
        next!(p; showvalues=generate_showvalues(report_value))
    end

    # next!(p; showvalues=generate_showvalues(report_value))
    # return report_value
    return
end

const LoopDict = Dict{String, Any}

function init_progress(loop_dict::LoopDict)
    loop_dict["ix_vals"] = 1
    loop_dict["ix_iter"] = 1

    p = Progress(loop_dict["n_iter"])
    return p
end

function show_val_unicode_plot(loop_dict::LoopDict)
    local_plot = lineplot(
        loop_dict["vals"],
    )
    show(local_plot)
    println("\n")
    return
end



function train_loop(
    model::HebbModel,
    data;
    n_vals::Integer = 100,
    n_epochs::Integer = 10,
    val_epoch::Bool = false,
)
    loop_dict = LoopDict()

    # Set up the epochs progress bar
    loop_dict["n_iter"] = if val_epoch
        length(data.train)
    else
        n_epochs
    end

    # Set up the validation intervals
    local_n_vals = min(n_vals, loop_dict["n_iter"])
    loop_dict["interval_vals"] = Int(floor(loop_dict["n_iter"] / local_n_vals))
    loop_dict["vals"] = zeros(Float32, local_n_vals)

    # Init the progress bar and loop tracking variables
    p = init_progress(loop_dict)

    # Iterate over each epoch
    for ie = 1:n_epochs
        # train_loader = Flux.DataLoader(data.train, batchsize=-1, shuffle=true)
        train_loader = Flux.DataLoader(data.train, batchsize=-1)
        if model.opts["gpu"]
            train_loader = train_loader |> gpu
        end

        # Iteratively train
        for (x, y) in train_loader
            if model.opts["immediate"]
                train_hebb_immediate(model, x, y)
            else
                train_hebb(model, x, y)
            end
            if val_epoch
                update_view_progress!(
                    p,
                    loop_dict,
                    model,
                    data,
                )
            end
        end

        # Compute validation performance
        if !val_epoch
            update_view_progress!(
                p,
                loop_dict,
                model,
                data,
            )
        else
            # Reset incrementers
            p = init_progress(loop_dict)

            local_plot = lineplot(
                loop_dict["vals"],
            )
            show(local_plot)
            println("\n")
        end
    end

    perf = test(model, data)
    @info "perf = $perf"
    return loop_dict["vals"]
end


function test(
    model::HebbModel,
    data::DeepART.DataSplit,
)
    n_test = length(data.test)

    y_hats = zeros(Int, n_test)
    test_loader = Flux.DataLoader(data.test, batchsize=-1)
    if model.opts["gpu"]
        y_hats = y_hats |> gpu
        test_loader = test_loader |> gpu
    end

    ix = 1
    for (x, _) in test_loader
        y_hats[ix] = argmax(model.model.chain(x))
        ix += 1
    end

    # y_hats = model(data.test.x |> gpu) |> cpu  # first row is prob. of true, second row p(false)
    # y_hats = argmaxmodel(data.test.x)  # first row is prob. of true, second row p(false)

    if model.opts["gpu"]
        y_hats = y_hats |> cpu
    end

    perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
    return perf
end

function profile_test(
    model,
    data,
    opts,
    n_epochs::Integer)
    _ = train_loop(
        model,
        data,
        n_epochs=n_epochs,
        eta=opts["eta"],
        beta_d=opts["beta_d"],
    )
end
