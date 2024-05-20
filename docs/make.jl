"""
    make.jl

# Description
This file builds the documentation for the `CFAR` project using Documenter.jl and other tools.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using
    Documenter,
    # DemoCards,
    Logging,
    Pkg

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Common variables of the script
PROJECT_NAME = "DeepART"
DOCS_NAME = "docs"

# Fix GR headless errors
ENV["GKSwstype"] = "100"

# Get the current workind directory's base name
current_dir = basename(pwd())
@info "Current directory is $(current_dir)"

# If using the CI method `julia --project=docs/ docs/make.jl`
#   or `julia --startup-file=no --project=docs/ docs/make.jl`
if occursin(PROJECT_NAME, current_dir)
    push!(LOAD_PATH, "../src/")
# Otherwise, we are already in the docs project and need to dev the above package
elseif occursin(DOCS_NAME, current_dir)
    Pkg.develop(path="..")
# Otherwise, building docs from the wrong path
else
    error("Unrecognized docs setup path")
end

# Inlude the local package
using DeepART

# using JSON
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end


# -----------------------------------------------------------------------------
# DOWNLOAD LARGE ASSETS
# -----------------------------------------------------------------------------

# Point to the raw FileStorage location on GitHub
top_url = raw"https://media.githubusercontent.com/media/AP6YC/FileStorage/main/DeepART/"

# List all of the files that we need to use in the docs
files = [
    "header.png",
]

# Make a destination for the files, accounting for when folder is AdaptiveResonance.jl
assets_folder = joinpath("src", "assets")
if basename(pwd()) == PROJECT_NAME || basename(pwd()) == PROJECT_NAME * ".jl"
    assets_folder = joinpath(DOCS_NAME, assets_folder)
end

download_folder = joinpath(assets_folder, "downloads")
mkpath(download_folder)
download_list = []

# Download the files one at a time
for file in files
    # Point to the correct file that we wish to download
    src_file = top_url * file * "?raw=true"
    # Point to the correct local destination file to download to
    dest_file = joinpath(download_folder, file)
    # Add the file to the list that we will append to assets
    push!(download_list, dest_file)
    # If the file isn't already here, download it
    if !isfile(dest_file)
        download(src_file, dest_file)
        @info "Downloaded $dest_file, isfile: $(isfile(dest_file))"
    else
        @info "File already exists: $dest_file"
    end
end

# Downloads debugging
detailed_logger = Logging.ConsoleLogger(stdout, Info, show_limited=false)
with_logger(detailed_logger) do
    @info "Current working directory is $(pwd())"
    @info "Assets folder is:" readdir(assets_folder, join=true)
    # full_download_folder = joinpath(pwd(), "src", "assets", "downloads")
    @info "Downloads folder exists: $(isdir(download_folder))"
    if isdir(download_folder)
        @info "Downloads folder contains:" readdir(download_folder, join=true)
    end
end

# -----------------------------------------------------------------------------
# GENERATE
# -----------------------------------------------------------------------------

# using Documenter
# using DeepART

assets = [
    joinpath("assets", "favicon.ico"),
]

makedocs(
    sitename = "DeepART",
    authors="Sasha Petrenko",
    # format = Documenter.HTML(),
    format=Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = assets,
        size_threshold = Int(1e6),
    ),
    pages=[
        "Home" => "index.md",
        "Guide" => "man/guide.md",
        "Internals" => [
            "Index" => "man/full-index.md",
            "Dev Index" => "man/dev-index.md",
            "Contributing" => "man/contributing.md",
        ],
    ],
    modules = [DeepART],
    repo = "https://github.com/AP6YC/DeepART/blob/{commit}{path}#L{line}",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

# -----------------------------------------------------------------------------
# DEPLOY
# -----------------------------------------------------------------------------

deploydocs(
    repo="github.com/AP6YC/DeepART.git",
    # devbranch="develop",
    devbranch="main",
    # push_preview = should_push_preview(),
)
