#= 
This file will contain tests to check the integration methods
=#
using Random, LinearAlgebra, Test, Distributions, PyPlot
include(abspath(@__DIR__, "../src/integrators.jl"))
include(abspath(@__DIR__,"../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__,"../src/actionComponents.jl"))
include(abspath(@__DIR__,"../src/interactions.jl"))

#======================= Leap forg tests ======================#

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

function Test_LeapFrogPQP_plot()
    rng = MersenneTwister()
    path_len = 10.0
    step_size = 0.01
    D = 1
    p_0 = rand(rng,(D))
    q_0 = ones(D)
    H(p,q) = sum(q.^2)/2 + sum(p.^2)/2
    pot(p) = sum(p.^2)/2
    dpdt(q) = q
    dqdt(p) = -p
    H_init = H(p_0,q_0)
    p_old,q_old = copy(p_0),copy(q_0)
    p,q, H_store, U_store, K_store = LeapFrogPQP_store(path_len, step_size, p_0, q_0, dqdt, pot)
    H_final = H(p,q)
    @test isapprox(H_init,H_final,atol=0.001)
    @test isapprox(H_store[1],H_store[end],atol=0.001)

    p_inverted,q_inverted, H_store, U_store, K_store  = LeapFrogPQP_store(path_len, step_size, p, -q, dqdt, pot)
    @test isapprox(H_store[1],H_store[end],atol=0.001)
    @test isapprox(p_inverted,p_old,atol=0.001)
    @test isapprox(q_inverted,-q_old,atol=0.001)
end

function Test_LeapFrogQPQ_plot()
    rng = MersenneTwister()
    path_len = 10.0
    step_size = 0.01
    D = 1
    p_0 = rand(rng,(D))
    q_0 = ones(D)
    H(p,q) = sum(q.^2)/2 + sum(p.^2)/2
    pot(p) = sum(p.^2)/2
    dpdt(q) = q
    dqdt(p) = -p
    H_init = H(p_0,q_0)
    p_old,q_old = copy(p_0),copy(q_0)
    p,q, H_store, U_store, K_store = LeapFrogQPQ_store(path_len, step_size, p_0, q_0, dqdt, pot)
    H_final = H(p,q)
    @test isapprox(H_init,H_final,atol=0.001)
    @test isapprox(H_store[1],H_store[end],atol=0.001)

    p_inverted,q_inverted, H_store, U_store, K_store  = LeapFrogQPQ_store(path_len, step_size, p, -q, dqdt, pot)
    @test isapprox(H_store[1],H_store[end],atol=0.001)
    @test isapprox(p_inverted,p_old,atol=0.001)
    @test isapprox(q_inverted,-q_old,atol=0.001)
end


function Test_SextonWeingarten()
    rng = MersenneTwister()
    path_len = 10.0
    step_size = 0.01
    D = 1
    p_0 = rand(rng,(D))
    q_0 = ones(D)
    H(p,q) = sum(q.^2)/2 + sum(p.^2)/2 + sum(p.^2)/2
    dpdt(q) = q
    dqdt_1(p) = -p
    dqdt_2(p) = -p
    H_init = H(p_0,q_0)
    p_old,q_old = copy(p_0),copy(q_0)
    p,q = SextonWeingartenIntegrator(path_len, step_size, p_0, q_0, dqdt_1, dqdt_2, 10)
    H_final = H(p,q)
    @test isapprox(H_init,H_final,atol=0.001)

    p_inverted,q_inverted = SextonWeingartenIntegrator(path_len, step_size, p, -q, dqdt_1, dqdt_2, 10)
    H_final = H(p,q)
    @test isapprox(p_inverted,p_old,atol=0.001)
    @test isapprox(q_inverted,-q_old,atol=0.001)
end


function Test_SextonWeingarten_plot()
    rng = MersenneTwister()
    path_len = 10.0
    step_size = 0.01
    D = 1
    p_0 = rand(rng,(D))
    q_0 = ones(D)
    H(p,q) = sum(q.^2)/2 + sum(p.^2)/2 + sum(p.^2)/2
    pot(p) = sum(p.^2)/2 + sum(p.^2)/2
    dpdt(q) = q
    dqdt_1(p) = -p
    dqdt_2(p) = -p
    H_init = H(p_0,q_0)
    p_old,q_old = copy(p_0),copy(q_0)
    p,q, H_store, U_store, K_store = SextonWeingartenIntegrator_store(path_len, step_size, p_0, q_0, dqdt_1, dqdt_2, pot, 10)
    H_final = H(p,q)
    @test isapprox(H_init,H_final,atol=0.001)
    @test isapprox(H_store[1],H_store[end],atol=0.001)
    clf()
    step = (0:1:Integer(ceil(path_len/step_size))-1).*step_size
    plot(step, real(H_store), label="energy")
    plot(step, real(K_store), label="kinetic")
    plot(step, real(U_store), label="potential")
    legend()
    savefig(abspath(@__DIR__, "../plots/integration_SextonWeingarten.png"))  
    
    p_inverted,q_inverted, H_store, U_store, K_store  = SextonWeingartenIntegrator_store(path_len, step_size, p, -q, dqdt_1, dqdt_2, pot, 10)
    @test isapprox(H_store[1],H_store[end],atol=0.001)
    @test isapprox(p_inverted,p_old,atol=0.001)
    @test isapprox(q_inverted,-q_old,atol=0.001)
end

#========================= integration for eq 35 ==========================#


function Test_Graphene_Integration_Mcomponent_35_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

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
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_Mcomponent_35(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_M(χ, M_function(ϕ), par ,lat) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_M_eq35(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ,π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_full_35(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_V(ϕ, V, par ,lat) + Action_M(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_V(ϕ, V, par, lat)-∇S_M_eq35(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_full_SextonWeingarten_35(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1, m=10)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_V(ϕ, V, par ,lat) + Action_M(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt_V(ϕ) = -∇S_V(ϕ, V, par, lat)
    dqdt_M(ϕ) = -∇S_M_eq35(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π = SextonWeingartenIntegrator(path_len, step_size, ϕ_0, π_0, dqdt_V, dqdt_M, m)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = SextonWeingartenIntegrator(path_len, step_size, ϕ, -π, dqdt_V, dqdt_M, m)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_full_35_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_V_cg(ϕ, V, par, lat)-∇S_M_eq35_cg(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 


function Test_Graphene_Integration_full_SextonWeingarten_35_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1, m=10)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    H(ϕ, π) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt_V(ϕ) = -∇S_V_cg(ϕ, V, par, lat)
    dqdt_M(ϕ) = -∇S_M_eq35_cg(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π = SextonWeingartenIntegrator(path_len, step_size, ϕ_0, π_0, dqdt_V, dqdt_M, m)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = SextonWeingartenIntegrator(path_len, step_size, ϕ, -π, dqdt_V, dqdt_M, m)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_Mcomponent_35_plot(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    
    Pot(ϕ) = Action_M(χ, M_function(ϕ), par ,lat)
    H(ϕ, π) = Pot(ϕ) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_M_eq35(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π, H_store, K_store ,U_store = LeapFrogQPQ_store(path_len, step_size, ϕ_0, π_0, dqdt, Pot)
    H_final = H(ϕ, π)
    clf()
    step = (0:1:Integer(ceil(path_len/step_size))-1).*step_size
    plot(step, real(H_store), label="energy")
    plot(step, real(K_store), label="kinetic")
    plot(step, real(U_store), label="potential")
    legend()
    savefig(abspath(@__DIR__, "../plots/integration_graphene_Fermionic_M_35.png"))
end 


function Test_Graphene_Integration_full_35_plot(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    M_function(ϕ) = FermionicMatrix_int_35(ϕ, V, par, lat)
    ρ = randn(rng, ComplexF64, lat.D)
    π_0 = rand(rng, Normal(), lat.D)
    χ = M_function(ϕ_0)*ρ
    Pot(ϕ) = Action_V(ϕ, V, par ,lat) + Action_M(χ, M_function(ϕ), par ,lat)
    H(ϕ, π) = Pot(ϕ) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_V(ϕ, V, par, lat)-∇S_M_eq35(ϕ, χ, M_function(ϕ), par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π, H_store, K_store ,U_store = LeapFrogQPQ_store(path_len, step_size, ϕ_0, π_0, dqdt, Pot)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    clf()
    step = (0:1:Integer(ceil(path_len/step_size))-1).*step_size
    plot(step, real(H_store), label="energy")
    plot(step, real(K_store), label="kinetic")
    plot(step, real(U_store), label="potential")
    legend()
    savefig(abspath(@__DIR__, "../plots/integration_graphene_Fermionic_full_35.png"))
end 
#======================== Integration for eq 41 =========================#
    function Test_Graphene_Integration_Mcomponent_41_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        H(ϕ, π) = Action_M_cg(χ, M_function(ϕ), par ,lat) + 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt(ϕ) = -∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ,π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        @show real(H_final)-real(H_init)
    
        ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
        @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
        @test isapprox(π_inverted,-π_old,atol=0.1)
    end 
    
    function Test_Graphene_Integration_Mcomponent_41(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        H(ϕ, π) = Action_M(χ, M_function(ϕ), par ,lat) + 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt(ϕ) = -∇S_M_eq41(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ,π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        @show real(H_final)-real(H_init)
    
        ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
        @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
        @test isapprox(π_inverted,-π_old,atol=0.1)
    end 
    
    function Test_Graphene_Integration_full_41(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        H(ϕ, π) = Action_V(ϕ, V, par ,lat) + Action_M(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt(ϕ) = -∇S_V(ϕ, V, par, lat)-∇S_M_eq41(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ, π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        @show real(H_final)-real(H_init)
    
        ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
        @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
        @test isapprox(π_inverted,-π_old,atol=0.1)
    end 

    function Test_Graphene_Integration_full_SextonWeingarten_41(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1, m=10)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        H(ϕ, π) = Action_V(ϕ, V, par ,lat) + Action_M(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt_V(ϕ) = -∇S_V(ϕ, V, par, lat)
        dqdt_M(ϕ) = -∇S_M_eq41(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ, π = SextonWeingartenIntegrator(path_len, step_size, ϕ_0, π_0, dqdt_V, dqdt_M, m)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        @show real(H_final)-real(H_init)
    
        ϕ_inverted, π_inverted = SextonWeingartenIntegrator(path_len, step_size, ϕ, -π, dqdt_V, dqdt_M, m)
        @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
        @test isapprox(π_inverted,-π_old,atol=0.1)
    end 
    
    function Test_Graphene_Integration_full_41_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        H(ϕ, π) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt(ϕ) = -∇S_V_cg(ϕ, V, par, lat)-∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ, π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        @show real(H_final)-real(H_init)
    
        ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
        @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
        @test isapprox(π_inverted,-π_old,atol=0.1)
    end 
    
    function Test_Graphene_Integration_full_SextonWeingarten_41_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1, m=10)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        H(ϕ, π) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)+ 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt_V(ϕ) = -∇S_V_cg(ϕ, V, par, lat)
        dqdt_M(ϕ) = -∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ, π = SextonWeingartenIntegrator(path_len, step_size, ϕ_0, π_0, dqdt_V, dqdt_M, m)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        @show real(H_final)-real(H_init)

        ϕ_inverted, π_inverted = SextonWeingartenIntegrator(path_len, step_size, ϕ, -π, dqdt_V, dqdt_M, m)
        @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
        @test isapprox(π_inverted,-π_old,atol=0.1)
    end 
    

    function Test_Graphene_Integration_Mcomponent_41_plot(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        
        Pot(ϕ) = Action_M(χ, M_function(ϕ), par ,lat)
        H(ϕ, π) = Pot(ϕ) + 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt(ϕ) = -∇S_M_eq41(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ, π, H_store, K_store ,U_store = LeapFrogQPQ_store(path_len, step_size, ϕ_0, π_0, dqdt, Pot)
        H_final = H(ϕ, π)
        clf()
        step = (0:1:Integer(ceil(path_len/step_size))-1).*step_size
        plot(step, real(H_store), label="energy")
        plot(step, real(K_store), label="kinetic")
        plot(step, real(U_store), label="potential")
        legend()
        savefig(abspath(@__DIR__, "../plots/integration_graphene_Fermionic_M_41.png"))
    end 
    
    
    function Test_Graphene_Integration_full_41_plot(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
        rng = MersenneTwister(1234)
    
        ϕ_0 = ones(lat.D)
        V = coulomb_potential(par, lat)
        M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)
        ρ = randn(rng, ComplexF64, lat.D)
        π_0 = rand(rng, Normal(), lat.D)
        χ = M_function(ϕ_0)*ρ
        Pot(ϕ) = Action_V(ϕ, V, par ,lat) + Action_M(χ, M_function(ϕ), par ,lat)
        H(ϕ, π) = Pot(ϕ) + 0.5*sum(π.*π)
        dpdt(π) = π
        dqdt(ϕ) = -∇S_V(ϕ, V, par, lat)-∇S_M_eq41(ϕ, χ, M_function(ϕ), par, lat)
        H_init = H(ϕ_0, π_0)
        ϕ_old,π_old = copy(ϕ_0),copy(π_0)
        ϕ, π, H_store, K_store ,U_store = LeapFrogQPQ_store(path_len, step_size, ϕ_0, π_0, dqdt, Pot)
        H_final = H(ϕ, π)
        @test isapprox(real(H_init), real(H_final), atol=atol)
        clf()
        step = (0:1:Integer(ceil(path_len/step_size))-1).*step_size
        plot(step, real(H_store), label="energy")
        plot(step, real(K_store), label="kinetic")
        plot(step, real(U_store), label="potential")
        legend()
        savefig(abspath(@__DIR__, "../plots/integration_graphene_Fermionic_full_41.png"))
    end 
    
#======================= potential part ==============================#

function Test_Graphene_Integration_PotentialComponent_plot(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)
    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    π_0 = rand(rng, Normal(), lat.D)
    Pot(ϕ) = Action_V_cg(ϕ, V, par ,lat)
    H(ϕ, π) = Pot(ϕ) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_V_cg(ϕ, V, par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π, H_store, K_store ,U_store = LeapFrogQPQ_store(path_len, step_size, ϕ_0, π_0, dqdt, Pot)
    H_final = H(ϕ, π)
    clf()
    step = (0:1:Integer(ceil(path_len/step_size))-1).*step_size
    plot(step, real(H_store), label="energy")
    plot(step, real(K_store), label="kinetic")
    plot(step, real(U_store), label="potential")
    legend()
    savefig(abspath(@__DIR__, "../plots/integration_graphene_potential.png"))
end 


function Test_Graphene_Integration_PotentialComponent(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    π_0 = rand(rng, Normal(), lat.D)
    H(ϕ, π) = Action_V(ϕ, V, par ,lat) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_V(ϕ, V, par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 

function Test_Graphene_Integration_PotentialComponent_cg(path_len, step_size, par::Parameters, lat::Lattice; atol=0.1)
    rng = MersenneTwister(1234)

    ϕ_0 = ones(lat.D)
    V = coulomb_potential(par, lat)
    π_0 = rand(rng, Normal(), lat.D)
    H(ϕ, π) = Action_V_cg(ϕ, V, par ,lat) + 0.5*sum(π.*π)
    dpdt(π) = π
    dqdt(ϕ) = -∇S_V_cg(ϕ, V, par, lat)
    H_init = H(ϕ_0, π_0)
    ϕ_old,π_old = copy(ϕ_0),copy(π_0)
    ϕ, π = LeapFrogQPQ(path_len, step_size, ϕ_0, π_0, dpdt, dqdt)
    H_final = H(ϕ, π)
    @test isapprox(real(H_init), real(H_final), atol=atol)
    @show real(H_final)-real(H_init)

    ϕ_inverted, π_inverted = LeapFrogQPQ(path_len, step_size, ϕ, -π, dpdt, dqdt)
    @test isapprox(ϕ_inverted,ϕ_old,atol=0.1)
    @test isapprox(π_inverted,-π_old,atol=0.1)
end 



Test_LeapFrogPQP()
Test_LeapFrogQPQ()
Test_SextonWeingarten()
Test_SextonWeingarten_plot()
Test_LeapFrogPQP_plot()
Test_LeapFrogQPQ_plot()

par = Parameters(2.0, 0.5, 1.0, 0.5)
lat = Lattice(6, 6, 16)
path_len = 10.0
step_size = 0.01
step_size_sw = 1.0
atol = 0.4#(roughly 67% acceptance)
m=10
# test for potential component of the action 
println("potential part ")
@time Test_Graphene_Integration_PotentialComponent(path_len, step_size, par, lat, atol=atol)
println("potential part cg ")
@time Test_Graphene_Integration_PotentialComponent_cg(path_len, step_size, par, lat, atol=atol)
Test_Graphene_Integration_PotentialComponent_plot(path_len, step_size, par,lat, atol=atol)

# tests for Fermionic matrix given by equation 35 
println("M 35 part ")
@time Test_Graphene_Integration_Mcomponent_35(path_len, step_size, par, lat, atol=atol)
println("M 35 part cg ")
@time Test_Graphene_Integration_Mcomponent_35_cg(path_len, step_size, par, lat, atol=atol)
Test_Graphene_Integration_Mcomponent_35_plot(path_len, step_size, par ,lat, atol=atol)
println("35 full ")
@time Test_Graphene_Integration_full_35(path_len, step_size, par, lat, atol=atol)
println("35 full cg ")
@time Test_Graphene_Integration_full_35_cg(path_len, step_size, par, lat, atol=atol)
println("35 full SW ")
@time Test_Graphene_Integration_full_SextonWeingarten_35(path_len, step_size_sw, par, lat, atol=atol,m=m)
println("35 full SW cg ")
@time Test_Graphene_Integration_full_SextonWeingarten_35_cg(path_len, step_size_sw, par, lat, atol=atol,m=m)
Test_Graphene_Integration_full_35_plot(path_len, step_size, par, lat, atol=atol)

# tests for Fermionic matrix given by equation 41
println("M 41 part ")
@time Test_Graphene_Integration_Mcomponent_41(path_len, step_size, par, lat, atol=atol)
println("M 41 part cg ")
@time Test_Graphene_Integration_Mcomponent_41_cg(path_len, step_size, par, lat, atol=atol)
Test_Graphene_Integration_Mcomponent_41_plot(path_len, step_size, par ,lat, atol=atol)
println("41 full ")
@time Test_Graphene_Integration_full_41(path_len, step_size, par, lat, atol=atol)
println("41 full cg ")
@time Test_Graphene_Integration_full_41_cg(path_len, step_size, par, lat, atol=atol)
println("41 full SW ")
@time Test_Graphene_Integration_full_SextonWeingarten_41(path_len, step_size_sw, par, lat, atol=atol,m=m)
println("41 full SW cg ")
@time Test_Graphene_Integration_full_SextonWeingarten_41_cg(path_len, step_size_sw, par, lat, atol=atol,m=m)
Test_Graphene_Integration_full_41_plot(path_len, step_size, par, lat, atol=atol)