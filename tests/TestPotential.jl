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
    r = distance_matrix(lat)./lat.a
    V_c = coulomb_potential(par, lat)
    V_p = partialScreenedCoulomb_potential(par, lat)
    V_s = shortScreenedCoulomb_potential(par, lat)
    clf()
    plot(vec(r),vec(V_c), ".",label="Std. Coulomb")
    plot(vec(r),vec(V_s),"o",label="ITEP screened ")
    plot(vec(r),vec(V_p), "*",label="Part. screened")
    legend()
    xlabel(L"$r_{ij}/a$")
    ylabel(L"$V(r)$")
    savefig(abspath(folder,"potentials.png"))
end 

function Test_potentialSymmetry(par::Parameters, lat::Lattice)
    folder=abspath(@__DIR__,"../plots")
    r = distance_matrix(lat)
    V_c = coulomb_potential(par, lat)
    V_p = partialScreenedCoulomb_potential(par, lat)
    V_s = shortScreenedCoulomb_potential(par, lat)
    @test V_c == transpose(V_c)
    @test V_p == transpose(V_p)
    @test V_s == transpose(V_s)
    invV_c = inv(V_c)
    invV_p = inv(V_p)
    invV_s = inv(V_s)
    @test isapprox(invV_c, transpose(invV_c))
    @test isapprox(invV_p, transpose(invV_p))
    @test isapprox(invV_s, transpose(invV_s))

end


par = Parameters(4.0, 0.0, 1.0, 0.5)
lat = Lattice(6, 6, 10)

Test_potentialPlotMatrix(par, lat)
Test_potentialPlot(par, lat)
Test_potentialSymmetry(par, lat)