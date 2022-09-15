#=This file will contain the computations of the greens function of the non-interacting graphene tight binding model.

The correlator will be computed in two ways. using the Fermionic matrix and using the analytical summation. 
We will also check both in k-space and position space keeping the temporal component the same.
=#

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianNoInteractions.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))


lat = Lattice(10, 18, 64)

ms = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
Δn_array = zeros(length(ms)[1])
for i in 1:length(ms)[1]
    par = Parameters(2.0, ms[i], 1.0, 0.5)
    M_no_int = FermionicMatrix_no_int(par, lat)
    Δn_array[i] = Δn(M_no_int, par, lat)
end

clf()
plot(ms, Δn_array, ".", label=L"$\alpha = 0.0$")
legend()
title(L"$\Delta n$ no interactions")
xlabel("m")
ylabel("Δn")
savefig(abspath(@__DIR__,"../plots/SublatticeSpin_no_int.png"))