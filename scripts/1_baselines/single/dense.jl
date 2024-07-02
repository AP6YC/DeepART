"""
    dense.jl

# Description
Single dense model experiment.
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

include("setup.jl")

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

@info "----------------- DENSE -----------------"

# Cluster:
# perf = 0.5794
# n_cat  = 228

# PC:
# perf = 0.5607
# n_cat=242

# Model definition
@info "Building model"
model = DeepART.get_rep_dense(n_input, head_dim)

@info "Building ART module"
art = DeepART.ARTINSTART(
    model,
    head_dim=head_dim,
    beta=params["beta_d"],
    beta_s=params["beta_s"],
    rho=0.6,
    # rho=0.3,
    # rho = 0.0,
    update="art",
    softwta=true,
    # gpu=GPU,
    gpu=false,
)

dev_xf = fdata.train.x[:, 1]
prs = Flux.params(art.model)
acts = Flux.activations(model, dev_xf)

# Train/test
begin
    debuglogger = ConsoleLogger(stdout, Logging.Info)
    Base.global_logger(debuglogger)

    @info "Beginning train-test loop"
    results = DeepART.tt_basic!(
        art,
        fdata,
        display=params["display"],
    )
    @info "Results: " results["perf"] results["n_cat"]
end

# Create the confusion matrix from this experiment
@info "Creating confusion matrix"
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    names,
    "dense_basic_confusion",
    params["exp_top"],
)
