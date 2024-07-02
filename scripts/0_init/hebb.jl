using Revise
using DeepART
using Flux
using ProgressMeter

data = DeepART.load_one_dataset(
    "iris",
    # n_train=1000,
    # n_test=500,
)

dev_x, dev_y = data.train[1]
n_input = size(dev_x)[1]
n_class = 4

model = Flux.@autosize (n_input,) Chain(
    Dense(
        _ => 10, sigmoid_fast, bias=false,
    ),
    Dense(
        _ => n_class, sigmoid_fast, bias=false,
    )
)

begin
    function train_hebb(chain, dev_x, dev_y)
        params = Flux.params(chain)
        acts = Flux.activations(chain, dev_x)
        ins = [dev_x, acts[1]]
        outs = [acts[1], acts[2]]

        outs[end] = zeros(Float32, size(outs[end]))
        outs[end][dev_y] = 1

        # @info "things:" ins outs

        # @info "sizes" length(params) size(params[1]) size(params[2]) size(ins[1]) size(ins[2]) size(outs[1]) size(outs[2])
        for ix = 1:2
            weights = params[ix]
            out = outs[ix]
            input = ins[ix]
            # @info "sizes:" size(weights) size(out) size(input)
            n_weight = size(weights)[1]
            for iw = 1:n_weight
                # weights[iw, :] .-= input .* out[iw] .* 0.1
                weights[iw, :] .= min.(input, weights[ix, :])
            end
            # weights .= instar(activation, weights, 0.1)
            # weights .+= input .* out .* 0.1
        end
    end

    old_weights = deepcopy(Flux.params(model))
    train_hebb(model, dev_x)
    new_weights = deepcopy(Flux.params(model))

    # @info sum(old_weights[1] - new_weights[1])
    # @info new_weights
    # @info "y:" dev_y argmax(model(dev_x))

    function train_loop(chain, data)
        n_train = length(data.train)
        @showprogress for ix = 1:n_train
            x, y = data.train[ix]
            train_hebb(chain, x, y)
        end
        # DeepART.get_perf(chain, data.test)
    end
end

train_loop(model, data)


# d_ws = torch.zeros(inputs.size(0))
# for idx, x in enumerate(inputs):
#     y = torch.dot(w, x)

#     d_w = torch.zeros(w.shape)
#     for i in range(y.shape[0]):
#         for j in range(x.shape[0]):
#             d_w[i, j] = self.c * x[j] * y[i]

#     d_ws[idx] = d_w

# return torch.mean(d_ws, dim=0)



# weights = Flux.params(model)

# weights[1]

# using DeepART

head_dim = 10
# model = DeepART.get_rep_fia_dense(n_input, head_dim)
# model = Flux.@autosize (n_input,) Chain(
#     DeepART.CC(),
#     # Dense(_, 512, sigmoid_fast, bias=false),
#     # DeepART.CC(),
#     Dense(_, 256, sigmoid_fast, bias=false),
#     DeepART.CC(),
#     Dense(_, head_dim, sigmoid_fast, bias=false),
# )
