"""
    epochs.jl

# Description
A convolutional experiment with multiple epochs.
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

size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

results = DeepART.tt_epochs!(
    art,
    data,
    display=DISPLAY,
    epochs=10,
)
@info "Results: " results["perf"] results["n_cat"]

# # Create the confusion matrix from this experiment
# DeepART.plot_confusion_matrix(
#     data.test.y,
#     results["y_hats"],
#     names,
#     "conv_basic_confusion",
#     EXP_TOP,
# )
