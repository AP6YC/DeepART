"""
    layers.jl
"""

@info """
\n####################################
###### LAYERS EXPERIMENT ######
####################################
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

@info "------- Loading dependencies -------"
using Revise
using Distributed

addprocs(16)

@everywhere begin

    using DeepART
    using DrWatson
    using Flux
    using Random
    using Plots
    using Random


    @info "------- Loading definitions -------"
    include("lib/lib.jl")

    @info "------- Loading Hebb module -------"
    import .Hebb

    n_add = 128

    outdir(args...) = DeepART.results_dir("layers", args...)
    mkpath(outdir())

    n_ext = 8
    n_rand = 25
end



for ix = 0:n_ext
    @distributed for rn_x = 1:n_rand

        # Construct the savename
        d = Dict(
            "ix" => ix,
            "rng" => rn_x,
        )
        local_savename = outdir(DrWatson.savename(d, "jld2"))

        # Skip if there is already a file
        if isfile(local_savename)
            @info "----- SKIPPING FILE $local_savename -----"
            continue
        end

        @info "------- Setting options -------"
        # opts = Hebb.load_opts("block-fuzzy.yml")
        opts = Hebb.load_opts("block-fuzzy-wh.yml")

        if ix > 0
            for jx = 1:ix
                push!(opts["block_opts"]["blocks"][1]["n_neurons"], n_add)
            end
        end
        @info opts["block_opts"]["blocks"][1]["n_neurons"]

        @info "------- Options post-processing -------"
        opts["sim_opts"]["rng_seed"] = rn_x
        Hebb.set_seed!(opts)

        @info "------- Loading dataset -------"
        data = Hebb.get_data(opts)

        model = Hebb.BlockNet(data, opts["block_opts"])

        vals = Hebb.train_loop(
            model,
            data,
            n_epochs = opts["sim_opts"]["n_epochs"],
            n_vals = opts["sim_opts"]["n_vals"],
            val_epoch = opts["sim_opts"]["val_epoch"],
            toshow=true,
        )

        perf = vals[end]

        dout = Dict(
            "ix" => ix,
            "rng" => rn_x,
            "perf" => perf,
            "vals" => vals,
        )
        DrWatson.tagsave(local_savename, dout)
        # DrWatson.safesave(saver, dout)
    end
end

# rmprocs(workers())

