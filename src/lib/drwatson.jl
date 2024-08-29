"""
    drwatson.jl

# Description
This file extends DrWatson workflow functionality such as by adding additional custom directory functions.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# CUSTOM DRWATSON DIRECTORY DEFINITIONS
# -----------------------------------------------------------------------------

"""
Points to the work directory containing raw datasets, processed datasets, and results.

$_ARG_DRWATSON
"""
function work_dir(args...)
    return projectdir("work", args...)
    # newdir(args...) = projectdir("work", args...)
    # # mkpath(newdir(args...))
    # return newdir(args...)
end

"""
Points to the results directory.

$_ARG_DRWATSON
"""
function results_dir(args...)
    return work_dir("results", args...)
    # newdir(args...) = work_dir("results", args...)
    # # mkpath(newdir(args...))
    # return newdir(args...)
end

"""
Points to the data directory.

$_ARG_DRWATSON
"""
function data_dir(args...)
    return work_dir("data", args...)
    # newdir(args...) = work_dir("data", args...)
    # # mkpath(newdir(args...))
    # return newdir(args...)
end

"""
Points to the configs directory.

$_ARG_DRWATSON
"""
function config_dir(args...)
    return work_dir("configs", args...)
    # newdir(args...) = work_dir("configs", args...)
    # # mkpath(newdir(args...))
    # return newdir(args...)
end

"""
`DrWatson`-style paper results directory.

$_ARG_DRWATSON
"""
function paper_results_dir(args...)
    local_hostname = gethostname()
    homedir = if local_hostname == "SASHA-XPS"
        "sap62"
    elseif local_hostname == "Sasha-PC"
        "Sasha"
    end
    @info homedir

    return joinpath(
        "C:\\",
        "Users",
        # "Sasha",
        homedir,
        "Dropbox",
        "Apps",
        "Overleaf",
        "Paper-DeepART",
        "images",
        "results",
        args...
    )
end

"""
`DrWatson`-style configs results directory.

$_ARG_DRWATSON
"""
function configs_dir(args...)
    # return work_dir("configs", args...)
    return results_dir("configs", args...)
    # Make the config folder for the experiment if it does not exist
    # mkpath(configs_dir())
end
