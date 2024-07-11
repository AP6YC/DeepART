using DelimitedFiles

Base.@kwdef mutable struct Lorenz
    dt::Float64 = 0.02
    σ::Float64 = 10
    ρ::Float64 = 28
    β::Float64 = 8 / 3
    x::Float64 = 1
    y::Float64 = 1
    z::Float64 = 1
end

function step!(l::Lorenz)
    dx = l.σ * (l.y - l.x)
    dy = l.x * (l.ρ - l.z) - l.y
    dz = l.x * l.y - l.β * l.z
    l.x += l.dt * dx
    l.y += l.dt * dy
    l.z += l.dt * dz
end

n_steps = 1500
lorenz = []
attractor = Lorenz()

for ix = 1:n_steps
    step!(attractor)
    push!(lorenz, [attractor.x, attractor.y, attractor.z])
end

open(joinpath("work", "results", "lorenz.txt"), "w") do io
    writedlm(io, l for l in lorenz)
end
