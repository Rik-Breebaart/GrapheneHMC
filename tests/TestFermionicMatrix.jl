#=
This file contains the tests for the fermionic matrices
=#
using Test, LinearAlgebra
include(abspath(@__DIR__,"../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__,"../src/hamiltonianNoInteractions.jl"))
include(abspath(@__DIR__,"../src/interactions.jl"))
include(abspath(@__DIR__,"../src/tools.jl"))


function Test_FermionicMatrixEq35_plot(par::Parameters, lat::Lattice)
    V = coulomb_potential(par, lat)
    ϕ = zeros(ComplexF64,(lat.D))
    ϕ .= 1:lat.D
    M = FermionicMatrix_int_35(ϕ, V, par, lat)
    plot_matrix(real(M),"FermionicMatrix_int_35_real")
    plot_matrix(imag(M),"FermionicMatrix_int_35_imag")

end

function Test_FermionicMatrixEq35_Hermiticity(par::Parameters, lat::Lattice)
    V = coulomb_potential(par, lat)
    ϕ = zeros(ComplexF64,(lat.D))
    ϕ .= 1:lat.D
    M = FermionicMatrix_int_35(ϕ, V, par, lat)
    @test ishermitian(M)==false
    @test ishermitian(M*adjoint(M))==true
end 

function Test_FermionicMatrixEq35_fast(par::Parameters, lat::Lattice)
    V = coulomb_potential(par, lat)
    ϕ = zeros(ComplexF64,(lat.D))
    ϕ .= 1:lat.D
    print("Time for saved part eq35:")
    @time M_saved_part = FermionicMatrix_int_35_saved_part(V, par, lat)
    @test ishermitian(M_saved_part*adjoint(M_saved_part))==true
    print("Time for phi part eq35:")
    @time M_quick = Fast_FermionicMatrix_int_35_phi_part(ϕ, M_saved_part, par, lat)
    @test ishermitian(M_quick)==false
    @test ishermitian(M_quick*adjoint(M_quick))==true
    print("Time for full eq35:")
    @time M_original = FermionicMatrix_int_35(ϕ, V, par, lat)    
    @test M_original==M_quick
end 

function Test_FermionicMatrixEq35_noInt(par::Parameters, lat::Lattice)
    V = zeros(Float64,(lat.dim_sub*2,lat.dim_sub*2))
    ϕ = zeros(ComplexF64,(lat.D))
    M_35 = FermionicMatrix_int_35(ϕ, V, par, lat)
    M_no_int = FermionicMatrix_no_int(par, lat)
    @test M_35==M_no_int    
end 

par = Parameters(4.0, 0.0, 1.0, 0.5)
lat = Lattice(4, 4, 10)

#tests for fermionic matrix build using equation 35.
Test_FermionicMatrixEq35_plot(par, lat)
Test_FermionicMatrixEq35_Hermiticity(par, lat)
Test_FermionicMatrixEq35_fast(par, lat)
Test_FermionicMatrixEq35_noInt(par, lat)

#tests for fermionic matrix build using equation 41