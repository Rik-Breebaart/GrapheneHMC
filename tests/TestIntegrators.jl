#= 
This file will contain tests to check the integration methods
=#
using Random, LinearAlgebra, Test, Distributions
include(abspath(@__DIR__, "../src/integrators.jl"))
include(abspath(@__DIR__,"../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__,"../src/actionComponents.jl"))
include(abspath(@__DIR__,"../src/interactions.jl"))


function Test_LeapFrogPQP()
    rng = MersenneTwister()
    path_len = 10.0
    step_size = 0.01
    D = 1
    p_0 = rand(rng,(D))
    q_0 = ones(D)
    H(p,q) = sum(q.^2)/2 + sum(p.^2)/2
    dpdt(q) = q
    dqdt(p) = -p
    H_init = H(p_0,q_0)
    p_old,q_old = copy(p_0),copy(q_0)
    p,q = LeapFrogPQP(path_len, step_size, p_0, q_0, dpdt, dqdt)
    H_final = H(p,q)
    @test isapprox(H_init,H_final,atol=0.001)

    p_inverted,q_inverted = LeapFrogPQP(path_len, step_size, p, -q, dpdt, dqdt)
    H_final = H(p,q)
    @test isapprox(p_inverted,p_old,atol=0.001)
    @test isapprox(q_inverted,-q_old,atol=0.001)
end

function Test_LeapFrogQPQ()
    rng = MersenneTwister()
    path_len = 10.0
    step_size = 0.01
    D = 1
    p_0 = rand(rng,(D))
    q_0 = ones(D)
    H(p,q) = sum(q.^2)/2 + sum(p.^2)/2
    dpdt(q) = q
    dqdt(p) = -p
    H_init = H(p_0,q_0)
    p_old,q_old = copy(p_0),copy(q_0)
    p,q = LeapFrogQPQ(path_len, step_size, p_0, q_0, dpdt, dqdt)
    H_final = H(p,q)
    @test isapprox(H_init,H_final,atol=0.001)

    p_inverted,q_inverted = LeapFrogQPQ(path_len, step_size, p, -q, dpdt, dqdt)
    H_final = H(p,q)
    @test isapprox(p_inverted,p_old,atol=0.001)
    @test isapprox(q_inverted,-q_old,atol=0.001)
end

function Test_Graphene_Integration_Mcomponent(par::Parameters, lat::Lattice)
    rng = MersenneTwister()
    path_len = 5.0
    step_size = 0.05

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_M_cg(χ, M_function(ϕ), par ,lat) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_M_eq35_cg(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ,π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=0.001)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_PotentialComponent(par::Parameters, lat::Lattice)
    rng = MersenneTwister()
    path_len = 5.0
    step_size = 0.01

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_M_cg(χ, M_function(ϕ), par ,lat) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_M_eq35_cg(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ,π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=0.001)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 


Test_LeapFrogPQP()
Test_LeapFrogQPQ()

par = Parameters(2.0, 0.0, 1.0, 0.5)
lat = Lattice(4, 4, 6)

Test_Graphene_Integration_Mcomponent(par, lat)
