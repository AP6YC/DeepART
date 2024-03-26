
# -----------------------------------------------------------------------------
# OLD TRAIN/TEST
# -----------------------------------------------------------------------------

# old_art = deepcopy(art)

# create_category!(art, xf, y_hat)
@showprogress for ix = 1:n_train
    xf = fdata.train.x[:, ix]
    label = data.train.y[ix]
    DeepART.train!(art, xf, y=label)
    # DeepART.train!(art, xf)
end

y_hats = Vector{Int}()
@showprogress for ix = 1:n_test
    xf = fdata.test.x[:, ix]
    y_hat = DeepART.classify(art, xf, get_bmu=true)
    push!(y_hats, y_hat)
end

perf = DeepART.ART.performance(y_hats, data.test.y[1:n_test])
@info "Perf: $perf, n_cats: $(art.n_categories), uniques: $(unique(y_hats))"

p = DeepART.create_confusion_heatmap(
    string.(collect(0:9)),
    data.test.y[1:n_test],
    y_hats,
)

# trainables = [weights[jx] for jx in [1, 3, 5]]
# ins = [acts[jx] for jx in [1, 3, 5]]
# outs = [acts[jx] for jx in [2, 4, 6]]
# for ix in eachindex(ins)
#     # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
#     trainables[ix] .+= DeepART.instar(ins[ix], outs[ix], trainables[ix], eta)
# end



# -----------------------------------------------------------------------------
# MERGEART
# -----------------------------------------------------------------------------

weights = [head[2].weight for head in art.heads]
# la = AdaptiveResonance.FuzzyART(
#     rho=0.1,
# )
la = AdaptiveResonance.DDVFA(
    rho_lb=0.1,
    rho_ub=0.3,
)
la.config = AdaptiveResonance.DataConfig(0, 1, art.opts.head_dim)

# for weight in weights
@showprogress for ix in eachindex(weights)
    weight = weights[ix]
    label = art.labels[ix]
    AdaptiveResonance.train!(la, weight, y=label)
end
@info "Before: $(art.n_categories), After: $(la.n_categories)"


# -----------------------------------------------------------------------------
# TRIM SINGLETONS
# -----------------------------------------------------------------------------

art2 = deepcopy(art)

DeepART.trimart!(art2)

y_hats2 = Vector{Int}()
@showprogress for ix = 1:n_test
    xf = fdata.test.x[:, ix]
    y_hat = DeepART.classify(art2, xf, get_bmu=true)
    push!(y_hats2, y_hat)
end
perf2 = DeepART.ART.performance(y_hats2, data.test.y[1:n_test])
@info "ORIGINAL: Perf: $(perf), n_cats: $(art.n_categories), uniques: $(unique(y_hats))"
@info "TRIMMED: Perf: $(perf2), n_cats: $(art2.n_categories), uniques: $(unique(y_hats2))"


# -----------------------------------------------------------------------------
# OLD MODELS
# -----------------------------------------------------------------------------

# model = @autosize (28, 28, 1, 1) Chain(
#     Conv((5,5),1=>6,relu),
#     Flux.flatten,
#     Dense(_=>15,relu),
#     Dense(15=>10,sigmoid),
#     softmax
# )

# size_tuple = (28, 28, 1, 1)

# Create a LeNet model
# model = @Flux.autosize (size_tuple,) Chain(
#     Conv((5,5),1 => 6, relu),
#     MaxPool((2,2)),
#     Conv((5,5),6 => 16, relu),
#     MaxPool((2,2)),
#     Flux.flatten,
#     Dense(256=>120,relu),
#     Dense(120=>84, relu),
#     Dense(84=>10, sigmoid),
#     softmax
# )

# for ix = 1:1000
#     x = reshape(data.train.x[:, :, ix], size_tuple)
#     acts = Flux.activations(model, x)
#     inputs = (xf, acts[1:end-1]...)
#     DeepART.instar(xf, acts, model, 0.0001)
# end

# ix = 1
# # x = fdata.train.x[:, ix]
# x = reshape(data.train.x[:, :, ix], size_tuple)
# y = data.train.y[ix]

# acts = Flux.activations(model, x)

# model = Chain(
#     Dense(n_input, 128, tanh),
#     Dense(128, 64, tanh),
#     Dense(64, n_classes, sigmoid),
#     # sigmoid,
#     # softmax,
# )

# model = Chain(
#     Dense(n_input*2, 128, tanh),
#     Dense(128, 64, tanh),
#     Dense(64, n_classes, sigmoid),
#     # sigmoid,
#     # softmax,
# )


# -----------------------------------------------------------------------------
# OLD CONV
# -----------------------------------------------------------------------------


# a, b, c = rand(3,2), rand(3,2), rand(3,2)
# a[1,1] = 0.1
# w = rand(3,2)
# d = mean(cat(min.(a, w), min.(b, w), min.(c, w), dims=3), dims=3)[:,:,1]
# e = min.(mean(cat(a, b, c, dims=3), dims=3)[:,:,1], w)

# d == e


# outs_conv = Flux.NNlib.fold(acts_conv[2][:,:,:,1], size(acts_conv[2]), size(prs_conv[1]))
# outs_conv = mean(acts_conv[1], dims=(1,2))

# function get_conv_inputs(image::Array, cdims)
#     # Convert the 2D image to 4D tensor with dimensions (height, width, channels, batch)
#     # image_4d = reshape(image, size(image)..., 1, 1)

#     col=similar(image, Flux.NNlib.im2col_dims(cdims))
#     # Use im2col to transform the image into columns
#     cols = Flux.NNlib.im2col!(col, image, cdims)

#     # Each column of 'cols' now represents a distinct receptive field of the convolution kernel
#     # Convert 'cols' back to a 2D array where each column is a vectorized receptive field
#     # conv_inputs = reshape(cols, :, size(cols, 4))
#     # return conv_inputs
#     return cols
# end

# xs = get_conv_inputs(dev_conv_x, (5,5,2,6))


# conv_model[1](dev_x)
# GPU && model |> gpu



# -----------------------------------------------------------------------------
# SFAM INSTART
# -----------------------------------------------------------------------------

# F2 layer size
# head_dim = 256
# head_dim = 512
# head_dim = 2048
head_dim = 1024

# Model definition
model = DeepART.get_rep_dense(n_input, head_dim)

art = DeepART.ARTINSTART(
    model,
    head_dim=head_dim,
    beta=0.01,
    rho=0.65,
    # rho=0.3,
    # epsilon=0.01,
    # rho=0.7,
    update="art",
    # head="fuzzy",
    softwta=true,
    # uncommitted=true,
    gpu=GPU,
)

# Train/test
results = DeepART.tt_basic!(art, fdata, n_train, n_test)
results = DeepART.tt_basic!(art, fdata, 2000, 2000)

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y[1:n_test],
    results["y_hats"],
    string.(collect(0:9)),
    "basic_artinstart",
    ["bp_instar"],
)
