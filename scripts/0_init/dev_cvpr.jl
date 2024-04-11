using Revise
using DeepART

HOSTNAME = gethostname()

data_dir = if HOSTNAME == "SASHA-XPS"
    joinpath("C:", "Users", "sap62", "Repos", "github", "DeepART", "work", "data", "indoorCVPR_09")
elseif HOSTNAME == "SASHA-PC"
    joinpath("E:", "dev", "data", "indoorCVPR_09")
end

data = DeepART.load_one_dataset(
    "isr",
    data_dir="data",
)

