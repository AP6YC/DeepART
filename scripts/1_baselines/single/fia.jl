using Revise
using Logging

include("setup.jl")

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

@info "----------------- FIA -----------------"

# Model definition
head_dim = 10
# model = DeepART.get_rep_fia_dense(n_input, head_dim)

size_tuple = (size(data.train.x)[1:3]..., 1)
model = DeepART.get_rep_fia_conv(size_tuple, head_dim)

art = DeepART.FIA(
    model,
    beta=BETA_D,
    rho=0.6,
    update="art",
    # update="instar",
    softwta=true,
    # gpu=GPU,
    gpu=false,
)

# old_weights = deepcopy(Flux.params(art.model[end-1])[1])
# old_weights = deepcopy(Flux.params(art.model))

# dev_xf = fdata.train.x[:, 1]
# prs = Flux.params(art.model)
# acts = Flux.activations(model, dev_xf)

# dev_xf = data.train.x[:, :, :, 1]
dev_xf, dev_y = data.train[2]
prs = Flux.params(art.model)
acts = Flux.activations(model, dev_xf)


begin
    debuglogger = ConsoleLogger(stdout, Logging.Info)
    Base.global_logger(debuglogger)

    # Train/test
    results = DeepART.tt_basic!(
        art,
        # fdata,
        data,
        display=DISPLAY,
        epochs=1,
    )
end

# acts = DeepART.learn_model(art, dev_xf, y=dev_y)
# debuglogger = SimpleLogger(Logging.Debug)

begin
    dev_xf, dev_y = data.train[4]
    debuglogger = ConsoleLogger(stdout, Logging.Debug)
    Base.global_logger(debuglogger)
    @debug "test"
    y_hat_train = DeepART.train!(art, dev_xf, y=dev_y)
    y_hat_test = DeepART.classify(art, dev_xf)
    # weights = Flux.params(art.model)
    # @info weights[length(weights)][:, 1]
    # @info weights
    acts = Flux.activations(model, dev_xf)
    @debug "y_hats: " dev_y y_hat_train y_hat_test
    @debug "acts: $(acts[end])"
end

begin
    check_weights = deepcopy(Flux.params(art.model))
    for p in check_weights
        result = p .+ 1
        p .= result
    end
    @info check_weights
end
# weights[length()]

# new_weights = Flux.params(art.model[end-1])[1]
# new_weights = deepcopy(Flux.params(art.model))
# weights .= DeepART.art_learn
# last_index = 3
# old_weights[last_index] - new_weights[last_index]

@info "Results: " results["perf"]

# # Create the confusion matrix from this experiment
# DeepART.plot_confusion_matrix(
#     data.test.y,
#     results["y_hats"],
#     names,
#     "dense_basic_confusion",
#     EXP_TOP,
# )
