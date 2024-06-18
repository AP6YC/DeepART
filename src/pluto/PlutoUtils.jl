"""
    pluto.jl

# Description
This is a set of extensions that are meant to be used in Pluto notebooks and with PlutoUI.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""


module PlutoUtils

using
    Markdown,
    Pluto,
    PlutoUI,
    DataStructures,
    NumericalTypeAliases,
    HypertextLiteral

import Base: show

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------

"""
Alias for the definition of a markdown string in Pluto notebooks.
"""
MDString = Markdown.MD

# -----------------------------------------------------------------------------
# COMMON DOCSTRINGS
# -----------------------------------------------------------------------------

"""
Common docstring, the arguments to functions taking a markdown string for display.
"""
const MD_ARG_STRING = """
# Arguments:
- `text::$MDString`: the markdown text to display in the box.
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Internal wrapper for displaying a markdown admonition.

# Arguments
- `type::String`: the type of admonition.
- `type::String`: the header of the admonition.
- `text::$MDString`: the markdown string to display in the box.
"""
function _admon(type::String, header::String, text::MDString)
    Markdown.MD(Markdown.Admonition(type, header, [text]))
end

"""
Shows a hint box in a Pluto notebook.

$MD_ARG_STRING
"""
function hint(text::MDString)
    _admon("hint", "Hint", text)
end
# hint(text::String) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

"""
Shows an danger box in a Pluto notebook.

$MD_ARG_STRING
"""
function keep_working(text::MDString=md"The answer is not quite right.")
    _admon("danger", "Keep working on it!", text)
end
# keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));

"""
Shows a correct box in a Pluto notebook.

$MD_ARG_STRING
"""
function correct(text::MDString=md"Great! You got the right answer! Let's move on to the next section.")
    _admon("correct", "Got it!", text)
end
# correct(text=md"Great! You got the right answer! Let's move on to the next section.") = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

"""
Shows an almost box in a Pluto notebook.

$MD_ARG_STRING
"""
function almost(text::MDString="The answer is almost correct!")
    _admon("warning", "Almost there!", text)
end
# almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

"""
Abstract supertype for parameter configurations.
"""
abstract type Config end

"""
A named container for elements of a PlutoUI `Slider`, which include the minimum, maximum, and default values.
"""
struct NumConfig{T} <: Config where T <: Real
    """
    The minimum value of the element.
    """
    min::T

    """
    The maximum value of the element.
    """
    max::T

    """
    The default value of the element.
    """
    default::T

    """
    The increment value for the element.
    """
    increment::T
end

"""
Constructor for [`NumConfig`](@ref) that assumes the `increment` value to be 1.
"""
function NumConfig(min::T, max::T, default::T) where T <: Real
    return NumConfig(min, max, default, one(T))
end

"""
A `Pluto` config option that uses a vector of strings as boolean options.
"""
struct OptConfig <: Config
    """
    A vector of the boolean option names as strings.
    """
    values::Vector{String}
end

"""
Generates the configuration slider widget.

# Arguments
- `dict::AbstractDict`: the config dictionary containing string keys mapping to [`Config`](@ref) elements.
"""
function config_input(dict::AbstractDict)

    # Run the combine() method on the  dictionary keys
    return PlutoUI.combine() do Child

        inputs = []
        for (key, value) in dict
            new_el = if value isa NumConfig
                md""" $(key): $(
                    Child(key, Slider(value.min:value.increment:value.max, default=value.default, show_value=true))
                )"""
                # md""" $(key): $(
                #     Child(key, Slider(value.min:value.increment:value.max, default=value.default, show_value=true))
                # )"""
            elseif value isa OptConfig
                md""" $(key): $(
                    Child(key, MultiCheckBox(value.values))
                )"""
            end

            push!(inputs, new_el)
        end

        # inputs = [
        #     md""" $(key): $(
        #         Child(key, Slider(value.min:value.max, default=value.default, show_value=true))
        #     )"""
        #     for (key, value) in dict
        # ]

        # Return the inputs as a Markdown string
        md"""
        $(inputs)
        """
    end
end

# function alt_config(dict::AbstractDict, Child)
#     inputs = []
#     # @info typeof(Child)
#     for (key, value) in dict

#         new_el = if value isa NumConfig
#             md""" $(key): $(
#                 Child(key, Slider(value.min:value.increment:value.max, default=value.default, show_value=true))
#             )"""
#         elseif value isa OptConfig
#             md""" $(key): $(
#                 Child(key, MultiCheckBox(value.values))
#             )"""
#         end
#         # @info typeof(new_el)
#         push!(inputs, new_el)
#     end

#     # @info inputs

#     return md"""
#     $(inputs)
#     """
# end

"""
Custom type for viewing matrices as grayscale images.

Taken from the following:
https://gist.github.com/pbouffard/3d48d3c47d9bd70e7c9f52f984d14245
"""
struct BWImage
    """
    The grayscale image data.
    """
    data::Array{UInt8, 2}

    """
    The display scaling parameter.
    """
    zoom::Int
end

"""
Constructor for a grayscale image from a data matrix for viewing.

# Arguments
- `data::Array{T, 2} where T <: Real`: the matrix for viewing.
- `zoom::Int=1`: optional, the display scaling parameter.
"""
function BWImage(data::Array{T, 2}; zoom::Int=1) where T <: Real
    BWImage(floor.(UInt8, clamp.(((data .- minimum(data)) / (maximum(data) .- minimum(data))) * 255, 0, 255)), zoom)
end

# https://gist.github.com/pbouffard/3d48d3c47d9bd70e7c9f52f984d14245
"""
A show overload for [`BWImage`](@ref) grayscale images for viewing.
"""
function show(io::IO, ::MIME"image/bmp", i::BWImage)

    orig_height, orig_width = size(i.data)
    height, width = (orig_height, orig_width) .* i.zoom
    datawidth = Integer(ceil(width / 4)) * 4

    bmp_header_size = 14
    dib_header_size = 40
    palette_size = 256 * 4
    data_size = datawidth * height * 1

    # BMP header
    write(io, 0x42, 0x4d)
    write(io, UInt32(bmp_header_size + dib_header_size + palette_size + data_size))
    write(io, 0x00, 0x00)
    write(io, 0x00, 0x00)
    write(io, UInt32(bmp_header_size + dib_header_size + palette_size))

    # DIB header
    write(io, UInt32(dib_header_size))
    write(io, Int32(width))
    write(io, Int32(-height))
    write(io, UInt16(1))
    write(io, UInt16(8))
    write(io, UInt32(0))
    write(io, UInt32(0))
    write(io, 0x12, 0x0b, 0x00, 0x00)
    write(io, 0x12, 0x0b, 0x00, 0x00)
    write(io, UInt32(0))
    write(io, UInt32(0))

    # color palette
    write(io, [[x, x, x, 0x00] for x in UInt8.(0:255)]...)

    # data
    padding = fill(0x00, datawidth - width)
    for y in 1:orig_height
        for z in 1:i.zoom
            line = vcat(fill.(i.data[y,:], (i.zoom,))...)
            write(io, line, padding)
        end
    end
end

"""
Helper function to visually display examples of digits that are confused with the target value.

# Arguments
- `data::DeepART.SupervisedDataset`:
- `y_hat::IntegerVector`:
- `truth::Integer`:
- `n_show::Integer`:
"""
function inspect_truth_errors(
    # data::DeepART.SupervisedDataset,
    data,
    y_hat::IntegerVector,
    selection::Integer,
    n_show::Integer,
)
    # Find the indices where the truth should be `truth`
    inds = findall(x -> x == selection+1, data.y[1:length(y_hat)])

    # Find all of those indices that resulted in an error
    errs = inds[findall(x -> data.y[x] != y_hat[x], inds)]

    # @info "lengths" length(inds) length(errs)
    if length(errs) == 0
        @info "No incorrect classifications of $(selection)"
        return
    end
    # Get the minimum number of elements for the number to show
    ln = min(length(errs), n_show)

    # Collect the images corresponding to the errors
    local_mats = [Matrix(transpose(data.x[:, :, errs[ix]])) for ix = 1:ln]
    ims = Gray.(local_mats[1])
    for ix = 2:ln
        ims = [ims Gray.(local_mats[ix])]
    end

    y_h = y_hat[errs[1:ln]] .- 1
    y_t = data.y[errs[1:ln]] .- 1

    @info "Images that were not correctly classified as $(selection):" y_h y_t ims
    return
end

"""
Helper function to visually display examples of digits that were incorrectly classified.

# Arguments
- `data::DeepART.SupervisedDataset`:
- `y_hat::IntegerVector`:
- `pred::Integer`:
- `n_show::Integer`:
"""
function inspect_prediction_errors(
    # data::DeepART.SupervisedDataset,
    data,
    y_hat::IntegerVector,
    selection::Integer,
    n_show::Integer,
    # truth::Integer
)
    # Find the indices where the predictions are pred
    inds = findall(x-> x == selection+1, y_hat)

    # Find the indices where the prediction should be pred
    # inds = findall(x-> x == pred, data.y[1:length(y_hat)])

    # Find all of those indices that resulted in an error
    errs = inds[findall(x -> data.y[x] != y_hat[x], inds)]

    # @info "lengths" length(inds) length(errs)
    if length(errs) == 0
        @info "No incorrect classifications of $(selection)"
        return
    end
    # Get the minimum number of elements for the number to show
    ln = min(length(errs), n_show)

    # Collect the images corresponding to the errors
    local_mats = [Matrix(transpose(data.x[:, :, errs[ix]])) for ix = 1:ln]
    ims = Gray.(local_mats[1])
    for ix = 2:ln
        ims = [ims Gray.(local_mats[ix])]
    end

    y_t = data.y[errs[1:ln]].-1
    y_h = y_hat[errs[1:ln]].-1

    @info "Images incorrectly predicted as $(selection):" y_h y_t ims
    return
end

end
