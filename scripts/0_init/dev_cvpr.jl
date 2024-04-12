"""
    dev_cvpr.jl

# Description
Development script for the Indoor Scene Recognition dataset.
"""

using Revise
using DeepART

HOSTNAME = gethostname()

data_dir = if HOSTNAME == "SASHA-XPS"
    joinpath("C:", "Users", "sap62", "Repos", "github", "DeepART", "work", "data", "indoorCVPR_09")
elseif HOSTNAME == "Sasha-PC"
    joinpath("E:", "dev", "data", "indoorCVPR_09")
elseif Sys.islinux()
    joinpath("lustre", "scratch", "sap625", "data", "indoorCVPR_09")
else
    error("Unknown hostname: $HOSTNAME")
end

data = DeepART.load_one_dataset(
    "isr",
    dir=data_dir,
)

data[3]

model = DeepART.get_rep_conv((64, 64, 3, 1), 15)

