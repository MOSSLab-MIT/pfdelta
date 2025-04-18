using Random, LinearAlgebra, Statistics, SpecialFunctions

# include("src/sample.jl")


function estimate_volume_montecarlo(A, b; num_samples=100_000)
    n = size(A, 2)

    # Estimate a bounding box (assume roughly within [-r, r]^n)
    r = maximum(abs.(b)) + 1.0
    box_volume = (2r)^n

    count_inside = 0
    for _ in 1:num_samples
        x = rand(n) .* (2r) .- r  # sample from [-r, r]^n
        if all(A * x .<= b)
            count_inside += 1
        end
    end

    volume_estimate = box_volume * count_inside / num_samples
    return volume_estimate
end


function estimate_volume_covariance(samples)
    n, N = size(samples)  # n = dimension, N = number of samples

    μ = mean(samples, dims=2)
    centered = samples .- μ
    covmat = centered * centered' / (N - 1)  # covariance matrix

    det_cov = det(covmat)
    if det_cov <= 0 || isnan(det_cov)
        error("Covariance matrix is degenerate or not full rank.")
    end

    # Volume of enclosing ellipsoid:
    #   V ≈ (π^(n/2) / Γ(n/2 + 1)) * sqrt(det(cov)) * k^n
    #   We scale by a factor `k` to stretch the ellipsoid to match the polytope (tune if needed)
    k = 2.0  # empirical scaling factor
    volume_estimate = (π^(n / 2) / gamma(n / 2 + 1)) * sqrt(det_cov) * k^n

    return volume_estimate
end

function estimate_polytope_volume_from_cprnd(A, b, x0; n_samples=10_000)
    samples = sample_polytope_cprnd(A, b, x0, n_samples)

    # Ensure samples are in shape (n, N)
    if size(samples, 1) != size(A, 2)
        samples = samples'  # transpose if rows are samples
    end

    return estimate_volume_covariance(samples)
end
