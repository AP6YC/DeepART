```@meta
DocTestSetup = quote
    using DeepART, Dates
end
```

![header](assets/downloads/header.png)

---

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

If you make use of this project, please generate your citation with the [CITATION.cff](https://github.com/AP6YC/DeepART/blob/main/CITATION.cff) file of the repository.
Alternatively, you may use the following BibTeX entry:

```bibtex
@software{Petrenko_AP6YC_DeepART_2024,
    author = {Petrenko, Sasha},
    doi = {10.5281/zenodo.10896042},
    month = jan,
    title = {{AP6YC/DeepART}},
    url = {https://github.com/AP6YC/DeepART},
    year = {2024}
}
```
