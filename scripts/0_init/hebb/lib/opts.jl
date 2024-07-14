"""
    opts.jl

# Description
Definitions for loading options.
"""

const TO_FLOAT32 = [
    "eta",
    "beta_d",
    "sigma",
]

# const FLUX_MAP = Dict(
#     "glorot_uniform" => Flux.glorot_uniform,
#     "sigmoid_fast" => Flux.sigmoid_fast,
#     "tanh_fast" => Flux.tanh_fast,
#     "relu" => Flux.relu,
# )

const TO_FLUX = [
    "init",
    "middle_activation",
]

function sanitize_opts!(opts)
    # Convert to Float32
    for key in keys(opts["model_opts"])
        if key in TO_FLOAT32
            opts["model_opts"][key] = Float32(opts["model_opts"][key])
        end
    end
    # opts["model_opts"]["eta"] = Float32(opts["model_opts"]["eta"])
    # opts["model_opts"]["beta_d"] = Float32(opts["model_opts"]["beta_d"])
    # opts["model_opts"]["sigma"] = Float32(opts["model_opts"]["sigma"])

    # Convert to Flux functions
    for key in keys(opts["model_opts"])
        if key in TO_FLUX
            # opts[key] = eval("Flux." * opts[key])
            opts["model_opts"][key] = getfield(Flux, Symbol(opts["model_opts"][key]))
        end
    end
end

function load_opts(name::AbstractString)
    opts = YAML.load_file(
        # joinpath("opts", name * ".yml");
        # joinpath(pwd(), "opts", name);
        joinpath(@__FILE__, "..", "..", "opts", name);
        dicttype=Dict{String, Any}
    )
    sanitize_opts!(opts)
    return opts
end
