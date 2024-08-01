"""
    opts.jl

# Description
Definitions for loading options.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

"""
Alias for the simulation options dictionary.
"""
const SimOpts = Dict{String, Any}
const ModelOpts = Dict{String, Any}

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
Key for the model options in the options dictionary.
"""
const MODEL_OPTS = "model_opts"

const OPTS_VERSION = "opts_version"

"""
Elements of model_opts that should be converted to Float32.
"""
const TO_FLOAT32 = [
    "eta",
    "beta_d",
    "sigma",
]

"""
Elements of model_opts that should be converted to Flux functions.
"""
const TO_FLUX = [
    "init",
    "middle_activation",
]

"""
All valid string options for the model.
"""
const SWITCHES = Dict{String, Any}(
    "model" => [
        "dense",
        "small_dense",
        "fuzzy",
        "conv",
        "fuzzy_new",
        "dense_new",
        "conv_new",
        "dense_spec",
        "fuzzy_spec",
        "widrow_hoff",
        "fuzzyartmap",
    ],

    "conv_strategy" => [
        "unfold",
        "patchwise",
    ],

    "learning_rule" => [
        "hebb",
        "oja",
        "instar",
        "fuzzyart",
    ],

    # Enumeration of beta rules.
    "beta_rule" => [
        "wta",
        "contrast",
        "softmax",
        "wavelet",
        "gaussian",
    ],
)

"""
All of the options that any one model must provide.
"""
const MODEL_DEPENDENCIES = Dict{String, Any}(
    "common" => [
        "index",
        "inputs",
        # "init",
        # "middle_activation",
        # "post_synaptic",
    ],
    "unique" => Dict{String, Any}(
        "dense" => [
            "beta",
            "beta_rule",
            "cc",
            "learning_rule",
            "init",
            "middle_activation",
            "post_synaptic",
        ],
        "fuzzy" => [
            "beta",
            "beta_rule",
            "cc",
            "learning_rule",
            "init",
            "middle_activation",
            "post_synaptic",
        ],
        "conv" => [
            "beta",
            "beta_rule",
            "conv_strategy",
            "cc",
            "learning_rule",
            "init",
            "middle_activation",
            "post_synaptic",
        ],
        "widrow_hoff" => [
            "eta",
            "final_sigmoid",
            "glorot_uniform",
        ],
        "fuzzyartmap" => [
            "rho",
            "beta_s",
        ],
    )
)


# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

function _sanitize_floats!(opts::ModelOpts)
    # Convert to Float32
    for key in keys(opts)
        if key in TO_FLOAT32
            opts[key] = Float32(opts[key])
        end
    end
    return
end

function _sanitize_flux!(opts::ModelOpts)
    # Convert to Flux functions
    for key in keys(opts)
        if key in TO_FLUX
            # opts[key] = eval("Flux." * opts[key])
            opts[key] = getfield(Flux, Symbol(opts[key]))
        end
    end
    return
end

function _sanitize_switches!(opts::ModelOpts)
    # Check for valid switches
    for key in keys(SWITCHES)
        # Check if the key is valid only if the switch is present
        if key in keys(opts) && !(opts[key] in SWITCHES[key])
            error("Invalid value for key $key: $(opts[key])")
        end
    end
    return
end

function _wavelet_correction(opts::ModelOpts)
    # Wavelet correction
    if haskey(opts, "beta_rule") && (opts["beta_rule"] == "wavelet")
        n_samples = 10000
        plot_range = -0.5

        x = range(-plot_range, plot_range, length=n_samples)
        y = Hebb.ricker_wavelet.(x, opts["sigma"])

        min_y = minimum(y)
        inds = findall(x -> x == min_y, y)
        # @info x[inds]
        opts["wavelet_offset"] = Float32(abs(x[inds][1]))
    end

    return
end

"""
V1: Sanitize the options dictionary.
"""
function _sanitize_opts_v1!(opts::SimOpts)
    _sanitize_floats!(opts[MODEL_OPTS])

    _sanitize_flux!(opts[MODEL_OPTS])

    _sanitize_switches!(opts[MODEL_OPTS])

    _wavelet_correction(opts[MODEL_OPTS])
end

function _sanitize_model_dependencies!(opts::ModelOpts)
    # Check for common dependencies
    for key in keys(MODEL_DEPENDENCIES["common"])
        if !(MODEL_DEPENDENCIES["common"][key] in keys(opts))
            error("Model dependency error: $(MODEL_DEPENDENCIES["common"][key])")
        end
    end

    # Check for model-specific dependencies
    for key in keys(MODEL_DEPENDENCIES["unique"])
        # Check if the model has unique dependencies
        if opts["model"] in MODEL_DEPENDENCIES["unique"][key]
            for dep in MODEL_DEPENDENCIES["unique"][key]
                if !(dep in keys(opts))
                    error("Model dependency error: $dep")
                end
            end
        end
    end
end

"""
Sanitize the options dictionary.
"""
function _sanitize_opts_v2!(opts::SimOpts)
    # for (_, value) in opts["block_opts"]["blocks"]
        # local_model_opts = value["model_opts"]
    for model_opts in opts["block_opts"]["blocks"]
        _sanitize_floats!(model_opts)
        _sanitize_flux!(model_opts)
        _sanitize_switches!(model_opts)
        _sanitize_model_dependencies!(model_opts)
        _wavelet_correction(model_opts)
    end
end

function load_opts(name::AbstractString)::SimOpts
    # Load the options
    opts = YAML.load_file(
        joinpath(@__FILE__, "..", "..", "opts", name);
        dicttype=SimOpts
    )

    if opts[OPTS_VERSION] == 1
        # Santize the options and do post-processing
        _sanitize_opts_v1!(opts)
    elseif opts[OPTS_VERSION] == 2
        _sanitize_opts_v2!(opts)
    else
        error("Invalid opts_version: $(opts[OPTS_VERSION])")
    end

    # Return the options
    return opts
end

"""
V1: simulation option getter.
"""
function _get_simopts_v1(opts::SimOpts)
    return opts
end

"""
V2: simulation option getter.
"""
function _get_simopts_v2(opts::SimOpts)
    return opts["sim_opts"]
end

"""
Get the simulation options, dispatching according to version.
"""
function get_simopts(opts::SimOpts)
    if opts[OPTS_VERSION] == 1
        return _get_simopts_v1(opts)
    elseif opts[OPTS_VERSION] == 2
        return _get_simopts_v2(opts)
    else
        error("Invalid opts_version: $(opts[OPTS_VERSION])")
    end
end

"""
V1: model option getter.
"""
function get_model_opts(opts::SimOpts)
    return get_simopts(opts)["model_opts"]
end

"""
V2: block option getter.
"""
function _get_block_opts(opts::SimOpts)
    return opts["block_opts"]
end

"""
V2: getter for the list of blocks.
"""
function _get_blocks(opts::SimOpts)
    return _get_block_opts(opts)["blocks"]
end

"""
V2: model option getter.
"""
function get_model_opts(opts::SimOpts, index::Integer)
    # return opts["block_opts"]["blocks"][index]["model_opts"]
    # return _get_blocks(opts)[index]["model_opts"]
    return _get_blocks(opts)[index]
end

"""
Set the seed for the simulation.
"""
function set_seed!(opts::SimOpts)
    Random.seed!(
        get_simopts(opts)["rng_seed"]
    )
    return
end
