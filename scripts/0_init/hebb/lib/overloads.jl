function incremental_supervised_train!(
    art::Hebb.HebbModel,
    x::RealArray,
    y::Integer,
)
    # y_hat = DeepART.train_hebb(art, x, y)
    y_hat = Hebb.train_hebb(art, x, y)
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y_hat
end


function incremental_supervised_train!(
    art::Hebb.BlockNet,
    x::RealArray,
    y::Integer,
)
    # y_hat = DeepART.train_hebb(art, x, y)
    y_hat = Hebb.train!(art, x, y)
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y_hat
end


function incremental_classify(
    art::Hebb.HebbModel,
    x::RealArray,
)
    # return DeepART.classify(art, x, get_bmu=true)
    return argmax(art.model.chain(x))
end

function incremental_classify(
    art::Hebb.BlockNet,
    x::RealArray,
)
    # return DeepART.classify(art, x, get_bmu=true)
    return Hebb.forward(art, x)
end
