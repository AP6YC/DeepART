"""
    conv.jl

# Description
Single convolutional experiment.
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

include("setup.jl")

# -----------------------------------------------------------------------------
# CONVOLUTIONAL
# -----------------------------------------------------------------------------

@info "----------------- CONVOLUTIONAL -----------------"

# perf = 0.7581
# n_cat  = 228

# PC:
# perf = 0.779
# n_cat = 124

@info "Building model"
size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

@info "Building ART module"
art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    # gpu=GPU,
    gpu=false,
)

begin
    debuglogger = ConsoleLogger(stdout, Logging.Info)
    Base.global_logger(debuglogger)

    @info "Beginning train-test loop"
    results = DeepART.tt_basic!(
        art,
        data,
        display=DISPLAY
    )
    @info "Results: " results["perf"] results["n_cat"]
end

# Create the confusion matrix from this experiment
@info "Creating confusion matrix"
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    names,
    "conv_basic_confusion",
    EXP_TOP,
)
