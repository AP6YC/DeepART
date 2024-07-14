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

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
Elements of "model_opts" that should be converted to Float32.
"""
const TO_FLOAT32 = [
    "eta",
    "beta_d",
    "sigma",
]

"""
Elements of "model_opts" that should be converted to Flux functions.
"""
const TO_FLUX = [
    "init",
    "middle_activation",
]

"""
Sanitize the options dictionary.
"""
function sanitize_opts!(opts::SimOpts)
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

"""
Load the options from a YAML file.
"""
function load_opts(name::AbstractString)::SimOpts
    opts = YAML.load_file(
        joinpath(@__FILE__, "..", "..", "opts", name);
        dicttype=Dict{String, Any}
    )
    sanitize_opts!(opts)
    return opts
end
