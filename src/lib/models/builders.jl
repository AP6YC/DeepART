"""
    builders.jl

# Description
Builders for ART modules from options.
"""

"""
Dispatcher for the ART module builders using a class-incremental split, build based upon from simply the first task.
"""
function get_module_from_options(
    d::AbstractDict,
    data::ClassIncrementalDataSplit,
)
    return get_module_from_options(
        d,
        data.train[1],
    )
end

"""
Dispatcher for building ART modules from options and a supervised dataset.
"""
function get_module_from_options(
    d::AbstractDict,
    # data::DataSplit,
    data::SupervisedDataset,
)
    n_input = size(data.x, 1)

    # Initialize the ART module
    art = if d["m"] == "SFAM"
        local_art = ART.SFAM(
            rho=d["rho"],
        )
        local_art.config = ART.DataConfig(0.0, 1.0, n_input)
        local_art
    elseif d["m"] == "DeepARTDense"
        # Model definition
        head_dim = d["head_dim"]
        # Model definition
        model = DeepART.get_rep_dense(n_input, head_dim)

        local_art = DeepART.ARTINSTART(
            model,
            head_dim=head_dim,
            beta=d["beta_d"],
            beta_s=d["beta_s"],
            rho=d["rho"],
            update="art",
            softwta=true,
            gpu=d["gpu"],
        )
        local_art
    elseif d["m"] == "DeepARTConv"
        # Model definition
        head_dim = d["head_dim"]

        # Get the size tuple instead of the input size for convolutions
        size_tuple = (size(data.x)[1:3]..., 1)
        conv_model = DeepART.get_rep_conv(size_tuple, head_dim)
        local_art = DeepART.ARTINSTART(
            conv_model,
            head_dim=head_dim,
            beta=d["beta_d"],
            beta_s=d["beta_s"],
            rho=d["rho"],
            update="art",
            softwta=true,
            gpu=d["gpu"],
        )

        local_art
    elseif d["m"] == "DeepARTDenseHebb"
        opts = Hebb.load_opts(config_dir("dense-fuzzy-dist.yml"))
        model = Hebb.HebbModel(data, opts["model_opts"])
        model
    elseif d["m"] == "DeepARTConvHebb"
        # opts = Hebb.load_opts(config_dir("lenet-dist.yml"))
        opts = Hebb.load_opts(config_dir("conv-fuzzy-dist.yml"))
        model = Hebb.HebbModel(data, opts["model_opts"])
        model
    elseif d["m"] == "DeepARTDenseBlock"
        # opts = Hebb.load_opts(config_dir("dense-fuzzy-dist.yml"))
        opts = Hebb.load_opts(config_dir("block-fuzzy-dist.yml"))
        model = Hebb.BlockNet(data, opts["block_opts"])
        model
    elseif d["m"] == "DeepARTConvBlock"
        # opts = Hebb.load_opts(config_dir("lenet-dist.yml"))
        opts = Hebb.load_opts(config_dir("block-conv-dist.yml"))
        model = Hebb.BlockNet(data, opts["block_opts"])
        model
    elseif d["m"] == "Oja"
        opts = Hebb.load_opts(config_dir("oja-dist.yml"))
        model = Hebb.HebbModel(data, opts["model_opts"])
        model
    elseif d["m"] == "Instar"
        opts = Hebb.load_opts(config_dir("instar-dist.yml"))
        model = Hebb.HebbModel(data, opts["model_opts"])
        model
    elseif d["m"] == "Contrast"
        opts = Hebb.load_opts(config_dir("contrast-dist.yml"))
        model = Hebb.HebbModel(data, opts["model_opts"])
        model
    else
        error("Unknown model: $(d["m"])")
    end

    return art
end