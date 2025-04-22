using LinearAlgebra, Plots

import Base: length

struct Grid{T <: Real}
    xvals::Vector{T}
    kvals::Vector{T}
end

length(G::Grid) = length(G.xvals)

function Grid(dx, N)
    L = dx * N
    xvals = dx .* collect(0:N-1)
    dk = 2π / L
    kvals = collect(-N÷2:N÷2 - 1) .* dk
    return Grid(xvals, kvals)
end

abstract type StateVector end

struct RealSpaceVector{T <: Complex, R <: Real} <: StateVector
    amps::Vector{T}
    grid::Grid{R}
end

struct ReciprocalSpaceVector{T <: Complex, R <: Real} <: StateVector
    amps::Vector{T}
    grid::Grid{R}
end

function RealSpaceVector(sv::ReciprocalSpaceVector)
    amps_k = sv.amps
    grid = sv.grid
    N = length(grid)
    amps_x = zeros(ComplexF64, N)

    for n in 1:N
        x = grid.xvals[n]
        for m in 1:N
            k = grid.kvals[m]
            amps_x[n] += amps_k[m] * exp(im * k * x)
        end
    end

    amps_x ./= √N
    return RealSpaceVector(amps_x, grid)
end

function ReciprocalSpaceVector(sv::RealSpaceVector)
    amps_x = sv.amps
    grid = sv.grid
    N = length(grid)
    amps_k = zeros(ComplexF64, N)

    for m in 1:N
        k = grid.kvals[m]
        for n in 1:N
            x = grid.xvals[n]
            amps_k[m] += amps_x[n] * exp(-im * k * x)
        end
    end

    amps_k ./= √N
    return ReciprocalSpaceVector(amps_k, grid)
end

LinearAlgebra.norm(V::StateVector) = norm(V.amps)

create_wvpx(G::Grid, x0, σ, p0) = RealSpaceVector(create_gaussian_wvp(G.xvals, x0, σ, p0), G)
create_wvpk(G::Grid, k0, σ, x0) = ReciprocalSpaceVector(create_gaussian_wvp(G.kvals, k0, σ, x0), G)

function create_gaussian_wvp(gridvalues, center, σ, phase)
    out = zeros(ComplexF64, length(gridvalues))

    for i in eachindex(out)
        v = gridvalues[i]
        out[i] = exp(-(v-center)^2 / (4*σ^2)) * exp(im*phase*v)
    end

    normalize!(out)

    return out
end

# Plotting support
function Plots.plot(sv::RealSpaceVector; kwargs...)
    x = sv.grid.xvals
    y = abs2.(sv.amps)
    plot(x, y; xlabel="Position", ylabel="|Amplitude|²", label="", kwargs...)
end

function Plots.plot(sv::ReciprocalSpaceVector; kwargs...)
    k = sv.grid.kvals
    y = abs2.(sv.amps)
    plot(k, y; xlabel="Wavevector", ylabel="|Amplitude|²", label="", kwargs...)
end


### Properties

# Average position for real-space state
function average_position(sv::RealSpaceVector; order::Int = 1)
    xvals = sv.grid.xvals
    probs = abs2.(sv.amps)
    return sum(xvals.^order .* probs)
end

# If in k-space, convert to x-space first
function average_position(sv::ReciprocalSpaceVector; order::Int = 1)
    return average_position(RealSpaceVector(sv); order=order)
end

# Average wavevector for k-space state
function average_wavevector(sv::ReciprocalSpaceVector; order::Int = 1)
    kvals = sv.grid.kvals
    probs = abs2.(sv.amps)
    return sum(kvals.^order .* probs)
end

# If in x-space, convert to k-space first
function average_wavevector(sv::RealSpaceVector; order::Int = 1)
    return average_wavevector(ReciprocalSpaceVector(sv); order=order)
end

function variance_position(sv::RealSpaceVector)
    μ = average_position(sv)
    μ2 = average_position(sv; order=2)
    return μ2 - μ^2
end

function variance_position(sv::ReciprocalSpaceVector)
    return variance_position(RealSpaceVector(sv))
end

function variance_wavevector(sv::ReciprocalSpaceVector)
    μ = average_wavevector(sv)
    μ2 = average_wavevector(sv; order=2)
    return μ2 - μ^2
end

function variance_wavevector(sv::RealSpaceVector)
    return variance_wavevector(ReciprocalSpaceVector(sv))
end