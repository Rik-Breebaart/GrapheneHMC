#=
This file contains the tests for the actions 
=#
using Test, LinearAlgebra
include(abspath(@__DIR__,"../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__,"../src/hamiltonianNoInteractions.jl"))
include(abspath(@__DIR__,"../src/actionComponents.jl"))
include(abspath(@__DIR__,"../src/interactions.jl"))
include(abspath(@__DIR__,"../src/tools.jl"))

function Test_action_non_interacting(par::Parameters, lat::Lattice)
    M = FermionicMatrix_no_int(par, lat)
    χ = ones(ComplexF64, (lat.D))
    @time S_1 = Action_M_cg(χ, M, par, lat)
    @time S_conf_1 = Action_M(χ, M, par, lat)
    @test isapprox(S_conf_1, S_1)
    @test isapprox(imag(S_1),0.0)
end 

function Test_action_interacting_35(par::Parameters, lat::Lattice)
    V = coulomb_potential(par, lat)
    ϕ = ones(ComplexF64,lat.D).*10
    M = FermionicMatrix_int_35(ϕ, V, par, lat)
    χ = ones(ComplexF64, (lat.D))
    @time S_1 = Action_M_cg(χ, M, par, lat)
    @time S_conf_1 = Action_M(χ, M, par, lat)
    @test isapprox(S_conf_1, S_1)
    @test isapprox(imag(S_1),0.0,atol=10^(-10))
end
    
function Test_action_interacting_41(par::Parameters, lat::Lattice)
    V = coulomb_potential(par, lat)
    ϕ = ones(ComplexF64,lat.D).*10
    M = FermionicMatrix_int_41(ϕ, par, lat)
    χ = ones(ComplexF64, (lat.D))
    @time S_1 = Action_M_cg(χ, M, par, lat)
    @time S_conf_1 = Action_M(χ, M, par, lat)
    @test isapprox(S_conf_1, S_1)
    @test isapprox(imag(S_1),0.0,atol=10^(-10))
end

par = Parameters(4.0, 0.0, 1.0, 0.5)
lat = Lattice(2, 2, 2)

Test_action_non_interacting(par, lat)
Test_action_interacting_35(par, lat)
Test_action_interacting_41(par, lat)