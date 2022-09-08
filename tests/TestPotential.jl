#= 
In this test file the potentials will be tested 
=# 

using Test, LinearAlgebra, PyPlot
include(abspath(@__DIR__,"../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__,"../src/hamiltonianNoInteractions.jl"))
include(abspath(@__DIR__,"../src/interactions.jl"))
include(abspath(@__DIR__,"../src/tools.jl"))


function Test_potentialPlotMatrix(par::Parameters, lat::Lattice)
    r = distance_matrix(lat)
    V_c = coulomb_potential(par, lat)
    V_p = partialScreenedCoulomb_potential(par, lat)
    V_s = shortScreenedCoulomb_potential(par, lat)
    plot_matrix(V_c, "coulomb")
    plot_matrix(V_p, "partial_screened_coulomb")
    plot_matrix(V_s, "screened_coulomb")
end 

function Test_potentialPlot(par::Parameters, lat::Lattice)
    folder=abspath(@__DIR__,"../plots")
    r = distance_matrix(lat)
    V_c = coulomb_potential(par, lat)
    V_p = partialScreenedCoulomb_potential(par, lat)
    V_s = shortScreenedCoulomb_potential(par, lat)
    clf()
    plot(vec(r),vec(V_c), ".",label="Coulomb")
    plot(vec(r),vec(V_s),"o",label="screened")
    plot(vec(r),vec(V_p), "*",label="Partial")
    legend()
    xlabel(L"$r_{ij}$")
    ylabel(L"$V/\epsilon$")
    savefig(abspath(folder,"potentials.png"))
end 


par = Parameters(4.0, 0.0, 1.0, 0.5)
lat = Lattice(4, 4, 20)

Test_potentialPlotMatrix(par, lat)
Test_potentialPlot(par, lat)