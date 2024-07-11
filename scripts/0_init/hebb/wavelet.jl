using Plots


function ricker_wavelet(t, sigma=1.0f0)
    # sigma = 1.0f0
    # sigma = 0.2f0
    return 2.0f0 / (sqrt(3.0f0 * sigma) * pi^(1.0f0 / 4.0f0)) * (1.0f0 - (t / sigma)^2) * exp(-t^2 / (2.0f0 * sigma^2))
end

n_samples = 1000
plot_range = 1.0
sigma = 0.2f0

x = range(-plot_range, plot_range, length=n_samples)
y = ricker_wavelet.(x, sigma)

# Ricker wavelet
plot(x, y,
    label="Ricker wavelet",
    xlabel="t",
    ylabel="phi(t)",
    title="Ricker",
    lw=2
)

# min_y = minimum(y)
# inds = findall(x -> x == min_y, y)
# x[inds]

# x = range(-5, 5, length=10)

# y = ricker_wavelet.(x)

# # Ricker wavelet
# plot(x, y,
#     label="Mexican hat wavelet",
#     xlabel="t",
#     ylabel="phi(t)",
#     title="Mexican hat wavelet",
#     lw=2
# )