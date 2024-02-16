using Documenter
using DeepART

makedocs(
    sitename = "DeepART",
    format = Documenter.HTML(),
    modules = [DeepART]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
