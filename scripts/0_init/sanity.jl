# This will prompt if neccessary to install everything, including CUDA:
using Flux, CUDA, Statistics, ProgressMeter

# # Sanity check
# model = Flux.@autosize (n_input,) Chain(
#     Dense(
#         _ => 10, sigmoid_fast, bias=false,
#     ),
#     Dense(
#         _ => 20, sigmoid_fast, bias=false,
#     ),
#     Dense(
#         _ => n_class, sigmoid_fast, bias=false,
#     )
# )

# for epoch in 1:1_000
#     Flux.train!(model, loader, optim) do m, x, y
#         y_hat = m(x)
#         Flux.logitcrossentropy(y_hat, y)
#     end
# end



opts = Dict{String, Any}(
    "gpu" => false,
    "profile" => false,
    "dataset" => "wine",
    # "n_train" => 1000,
    # "n_test" => 100,
    # "flatten" => true,
    "n_epochs" => 1000,
    "eta" => 1.0,
    "beta_d" => 1.0,
    "rng_seed" => 1234,
)

data = DeepART.load_one_dataset(
    # "iris",
    "wine",
    # n_train=opts["n_train"],
    # n_test=opts["n_test"],
    # flatten=opts["flatten"],
)


function gradient_descent(data)
    # Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
    # noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
    # truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

    noisy = data.train.x
    truth = data.train.y

    n_input = size(dev_x)[1]
    n_class = length(unique(data.train.y))

    # Define our model, a multi-layer perceptron with one hidden layer of size 3:
    # model = Chain(
    #     Dense(2 => 3, tanh),   # activation function inside layer
    #     BatchNorm(3),
    #     Dense(3 => 2)) |> gpu        # move model to GPU, if available

    # Sanity check
    model = Flux.@autosize (n_input,) Chain(
        Dense(_ => 40, sigmoid_fast, bias=false),
        Dense(_ => 20, sigmoid_fast, bias=false),
        Dense(_ => 10, sigmoid_fast, bias=false),
        Dense(_ => n_class, sigmoid_fast, bias=false),
    )
    @info model
    model = model |> gpu

    # The model encapsulates parameters, randomly initialised. Its initial output is:
    out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}
    probs1 = softmax(out1)      # normalise to get probabilities

    # To train the model, we use batches of 64 samples, and one-hot encoding:
    # target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
    target = Flux.onehotbatch(truth, unique(truth))                   # 2×1000 OneHotMatrix
    loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);
    # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

    optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

    # Training loop, using the whole data set 1000 times:
    losses = []
    @showprogress for _ in 1:opts["n_epochs"]
        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                # @info y_hat y
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
    end

    optim # parameters, momenta and output have all changed
    noisy = data.test.x
    truth = data.test.y

    out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)
    probs2 = softmax(out2)      # normalise to get probabilities
    mean((probs2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!


    @info "------- Defining test -------"
    function test_sanity(model, data)
        n_test = length(data.test)

        y_hats = zeros(Int, n_test)
        test_loader = Flux.DataLoader(data.test, batchsize=-1)

        ix = 1
        for (x, _) in test_loader
            y_hats[ix] = argmax(model(x |> gpu) |> cpu)
            ix += 1
        end

        perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
        return perf
    end

    perf = test_sanity(model, data)

end

perf = gradient_descent(data)
@info "Sanity check performance:" perf