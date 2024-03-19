"""
Common code for custom Flux.jl layer definitions.
"""

"""
Returns the complement code of the input
"""
function complement_code(x)
    vcat(x, 1.0 .- x)
end
