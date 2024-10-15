function hebb_preprocess(
    art::Hebb.HebbModel,
    x::RealArray,
)
    # local_x = if (art.opts["model"] == "DeepARTConv") || (art.opts["model"] == "DeepARTConvHebb") && (length(size(x)) == 3)
    # @info (art.opts["model"] == "conv")
    # @info (art.opts["model"] == "conv_new")
    # @info (length(size(x)) == 3)
    local_x = if ((art.opts["model"] == "conv") || (art.opts["model"] == "conv_new")) && (length(size(x)) == 3)
    # @info art.opts["model"]
    # local_x = if ((art.opts["model"] == "conv") || (art.opts["model"] == "conv_new"))
        reshape(x, size(x)..., 1)
    else
        vec(x)
    end
    # @info size(local_x)

    return local_x
end

function block_preprocess(
    art::Hebb.BlockNet,
    x::RealArray,
)
    # local_x = if art.layers[1] isa Hebb. && (length(size(x)) == 3)
    local_x = if art.opts["blocks"][1]["model"] == "lenet" && (length(size(x)) == 3)
        reshape(x, size(x)..., 1)
    else
        vec(x)
    end
    # @info size(local_x)

    return local_x
end

function incremental_supervised_train!(
    art::Hebb.HebbModel,
    x::RealArray,
    y::Integer,
)
    local_x = hebb_preprocess(art, x)
    # @info "inside inc: " size(local_x)
    # y_hat = DeepART.train_hebb(art, x, y)
    # y_hat = Hebb.train_hebb(art, x, y)
    y_hat = argmax(vec(Hebb.train_hebb(art, local_x, y)))
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y_hat
end


function incremental_supervised_train!(
    art::Hebb.BlockNet,
    x::RealArray,
    y::Integer,
)
    # y_hat = DeepART.train_hebb(art, x, y)

    local_x = block_preprocess(art, x)

    # y_hat = Hebb.train!(art, x, y)
    y_hat = argmax(vec(Hebb.train!(art, local_x, y)))
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y_hat
end


function incremental_classify(
    art::Hebb.HebbModel,
    x::RealArray,
)
    # return DeepART.classify(art, x, get_bmu=true)
    local_x = hebb_preprocess(art, x)

    # return argmax(art.model.chain(x))
    return argmax(vec(art.model.chain(local_x)))
end

function incremental_classify(
    art::Hebb.BlockNet,
    x::RealArray,
)
    local_x = block_preprocess(art, x)

    # return DeepART.classify(art, x, get_bmu=true)
    # return Hebb.forward(art, x)
    # return Hebb.forward(art, local_x)
    return argmax(vec(Hebb.forward(art, local_x)))
end
