include("setup.jl")

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

@info "----------------- FIA -----------------"

# Model definition
head_dim = 10
model = DeepART.get_rep_fia_dense(n_input, head_dim)

art = DeepART.FIA(
    model,
    beta=BETA_D,
    rho=0.6,
    update="art",
    softwta=true,
    # gpu=GPU,
    gpu=false,
)

dev_xf = fdata.train.x[:, 1]
prs = Flux.params(art.model)
acts = Flux.activations(model, dev_xf)

# Train/test
results = DeepART.tt_basic!(
    art,
    fdata,
    display=DISPLAY,
)
# @info "Results: " results["perf"] results["n_cat"]
@info "Results: " results["perf"]

# # Create the confusion matrix from this experiment
# DeepART.plot_confusion_matrix(
#     data.test.y,
#     results["y_hats"],
#     names,
#     "dense_basic_confusion",
#     EXP_TOP,
# )
