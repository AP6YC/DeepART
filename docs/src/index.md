```@meta
DocTestSetup = quote
    using DeepART, Dates
end
```

# DeepART.jl

Documentation for the `DeepART.jl` project.

See the [Index](@ref main-index) for the complete list of documented functions and types.

## Manual Outline

This documentation is split into the following sections:

```@contents
Pages = [
    "man/guide.md",
    "../examples/index.md",
    "man/modules.md",
    "man/contributing.md",
    "man/full-index.md",
    "man/dev-index.md",
]
Depth = 1
```

## Documentation Build

This documentation was built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) with the following version and OS:

```@example
using DeepART, Dates # hide
println("DeepART v$(DEEPART_VERSION) docs built $(Dates.now()) with Julia $(VERSION) on $(Sys.KERNEL)") # hide
```

## Citation

If you make use of this project, please generate your citation with the [CITATION.cff](../../CITATION.cff) file of the repository.
Alternatively, you may use the following BibTeX entry for the JOSS paper associated with the repository:
