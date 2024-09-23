# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using Flux
using BSON
using CUDA
using StatsBase: mean
using BenchmarkTools
using Tullio

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

CUDA.functional()
device = Flux.get_device(; verbose=true)

# data = randn(32, 32, 3, 100)
data = randn(Float32, 32, 32, 1, 100)

model = Chain(
    Conv((3, 3), 1=>16, relu),
)

model = model |> gpu
data = data |> gpu

a = model(data)
a = a |> cpu

# -----------------------------------------------------------------------------
# Global
# -----------------------------------------------------------------------------

index = 1
weights = model[1].weight

full_size = size(weights)
n_kernels = full_size[4]
kernel_shape = full_size[1:3]

x = data[:, :, :, [index]]
out = model(x)

unfolded = Flux.NNlib.unfold(
    x,
    full_size,
    # pad=(2,2),
)

n_windows = size(unfolded, 1)

flat_out = reshape(out, n_windows, n_kernels)
# flat_weights = reshape(weights, :, n_kernels)'
flat_weights = reshape(weights, :, n_kernels)

for ix = 1:n_windows
    local_in = unfolded[ix, :]
    local_out = flat_out[ix, :]
    local_weight = flat_weights
    # beta = mean(local_out)
    beta = 0.1
    jx = argmax(local_out)
    local_weight[:, jx] += beta * min.(local_in, local_weight[:, jx]) + local_weight[:, jx] * (1 - beta)
end

# jxs = argmax(flat_out, dims=2)
# # repeat_weights = repeat(flat_weights, n_windows, 1)
# beta = 0.1
# # flat_weights += beta * min.(unfolded, flat_weights[jxs]) + flat_weights[jxs] * (1 - beta)
# for ix = 1:n_windows

# end



# -----------------------------------------------------------------------------
# Functions and Benchmarks
# -----------------------------------------------------------------------------




function train!(model, data, index)
    weights = model[1].weight

    full_size = size(weights)
    n_kernels = full_size[4]
    kernel_shape = full_size[1:3]

    x = data[:, :, :, [index]]
    out = model(x)

    unfolded = Flux.NNlib.unfold(
        x,
        full_size,
        # pad=(2,2),
    )

    n_windows = size(unfolded, 1)

    flat_out = reshape(out, n_windows, n_kernels)
    # flat_weights = reshape(weights, :, n_kernels)'
    flat_weights = reshape(weights, :, n_kernels)

    for ix = 1:n_windows
        local_in = unfolded[ix, :]
        local_out = flat_out[ix, :]
        local_weight = flat_weights
        # beta = mean(local_out)
        beta = 0.1
        jx = argmax(local_out)
        local_weight[:, jx] += beta * min.(local_in, local_weight[:, jx]) + local_weight[:, jx] * (1 - beta)
    end

    # jxs = argmax(flat_out, dims=2)
    # beta = 0.1
    # flat_weights += beta * min.(unfolded, flat_weights[jxs]) + flat_weights[jxs] * (1 - beta)
end

function train!(model, data)
    for ix = 1:size(data)[end]
        train!(model, data, ix)
    end
end

model = model |> gpu
data = data |> gpu

@time train!(model, data)
CUDA.@profile train!(model, data)
CUDA.@profile train!(model, data, 1)

model = model |> cpu
data = data |> cpu

@time train!(model, data)


# -----------------------------------------------------------------------------
# Tullio
# -----------------------------------------------------------------------------


A = rand(3, 10, 20, 5)
findmax4Dt(A) = @tullio (fun) out[i,j,k] := (z, A[i,j,k,z])  init = (0,-Inf);
fun((i,x), (j,y)) = ifelse(x>y, (i,x), (j,y));

@btime first.(findmax4Dt($A));

function whichmax4Dt(A)
    fun((i,x), (j,y)) = ifelse(x>y, (i,x), (j,y))
    out = similar(A, Int, axes(A)[1:end-1])
    @tullio (fun) out[i,j,k] = first <| (z, A[i,j,k,z])  init = (0,-Inf)  threads=false
end;

@btime whichmax4Dt($A);

using Tullio
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

using Tracker # or Zygote
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CUDA, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

# full_size = size(weights)
# n_kernels = full_size[4]
# kernel_shape = full_size[1:3]

# unfolded = Flux.NNlib.unfold(data, full_size)

# local_in = reshape(mean(reshape(unfolded, :, kernel_shape...), dims=1), :)

# # Get the averaged and reshaped local output
# local_out = reshape(mean(out, dims=(1, 2)), n_kernels)

# # Reshape the weights to be (n_kernels, n_features)
# local_weight = reshape(weights, :, n_kernels)'