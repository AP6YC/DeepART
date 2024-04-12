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
model = DeepART.get_rep_dense(n_input, head_dim)

art = DeepART.ARTINSTART(
    model,
    head_dim=head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
    rho=0.6,
    # rho=0.3,
    # rho = 0.0,
    update="art",
    softwta=true,
    gpu=GPU,
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
@info "Results: " results["perf"] results["n_cat"]

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    names,
    "dense_basic_confusion",
    EXP_TOP,
)