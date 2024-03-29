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


lat = Lattice(6, 6, 16)
lat_analytic = Lattice(lat.Lm, lat.Ln, 256)

par = Parameters(2.0, 0.5, 1.0, 0.5)

int(x) = floor(Int, x)
particle_x = Particle(1, 1, 0, 1)
particle_y = Particle(1, 1, 0, 1)

correlator_continuum = greensFunctionGraphene_spatial(particle_x,particle_y, par, lat_analytic)
correlator = greensFunctionGraphene_spatial(particle_x,particle_y, par, lat)
M_no_int = FermionicMatrix_no_int(par, lat)
correlator_M = greens_function_spatial(M_no_int, particle_x, particle_y, par, lat)
name = ["A","B"]
name_G = [L"G_{AA}" L"G_{AB}" 
        L"G_{BA}"  L"G_{BB}"]
τ = (0:1:lat.Nt-1).*(par.β/lat.Nt)
τ_analytic = (0:1:lat_analytic.Nt-1).*(par.β/lat_analytic.Nt)
for i=[1,2]
    for j = [1,2]
        clf()
        if i==j
            semilogy(τ_analytic, real.(correlator_continuum[:,i,j]), label="Continuum limit")
            semilogy(τ, real.(correlator[:,i,j]), "*", label="analytical discrete")
            semilogy(τ, real.(correlator_M[:,i,j]),".", label="Fermionic Matrix")
        else 
            plot(τ_analytic, real.(correlator_continuum[:,i,j]), label="Continuum limit")
            plot(τ, real.(correlator[:,i,j]), "*", label="analytical discrete")
            plot(τ, real.(correlator_M[:,i,j]),".", label="Fermionic Matrix")
        end
        legend()
        xlabel(L"time")
        ylabel(string(L"\langle ",name_G[i,j],L"(τ) \rangle"))
        savefig(abspath(@__DIR__, string("../plots/G_",lat.Lm,"_",lat.Ln,"_",lat.Nt,"_",name[i],name[j])))
    end 
end
# check = zeros((2,2))
# for i=[1,2], j = [1,2]
#     check[i,j] = mean(real.(correlator[:,i,j])./real.(correlator_M[:,i,j]))
# end
# @show check



k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
k_b = (4*pi)/(3*lat.a)* [0,1]
ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b
correlator_momentum_continuum = greensFunctionGraphene_kspace(ks(int(lat.Lm/2),int(lat.Ln/2)), par, lat_analytic)
correlator_momentum = greensFunctionGraphene_kspace(ks(int(lat.Lm/2),int(lat.Ln/2)), par, lat)
correlator_M_momentum = greens_function_kspace(M_no_int, ks(int(lat.Lm/2),int(lat.Ln/2)), par, lat)
for i=[1,2]
    for j = [1,2]
        clf()
        if i==j
            semilogy(τ_analytic, real.(correlator_momentum_continuum[:,i,j]), label="Continuum limit")
            semilogy(τ, real.(correlator_momentum[:,i,j]), "*", label="analytical discrete")
            semilogy(τ, real.(correlator_M_momentum[:,i,j]),".", label="Fermionic Matrix")
        else 
            plot(τ_analytic, real.(correlator_momentum_continuum[:,i,j]), label="Continuum limit")
            plot(τ, real.(correlator_momentum[:,i,j]), "*", label="analytical discrete")
            plot(τ, real.(correlator_M_momentum[:,i,j]),".", label="Fermionic Matrix")
        end
        legend()
        xlabel(L"time")
        ylabel(string(L"\langle ",name_G[i,j],L"(τ,k) \rangle"))
        savefig(abspath(@__DIR__, string("../plots/G_",lat.Lm,"_",lat.Ln,"_",lat.Nt,"_k",name[i],name[j])))
    end 
end

# check = zeros((2,2))
# for i=[1,2], j = [1,2]
#     check[i,j] = mean(real.(correlator_momentum[:,i,j])./real.(correlator_M_momentum[:,i,j]))
# end
# @show check



C_pm(correlator,q) = (correlator[:,1,1]+correlator[:,2,2]+q.*(correlator[:,1,2]+correlator[:,2,1]))./2

c_M_plus = C_pm(correlator_M_momentum,1)
c_M_min = C_pm(correlator_M_momentum,-1)
c_plus = C_pm(correlator_momentum,1)
c_min = C_pm(correlator_momentum,-1)

clf()
semilogy(τ, real.(c_M_plus), ".", label=L"Fermionic G_{+}")
semilogy(τ_analytic, real.(c_plus), "*", label=L"analytical G_{+}")
semilogy(τ, real.(c_M_min), ".", label=L"Fermionic G_{-}")
semilogy(τ_analytic, real.(c_min), "*", label=L"analytical G_{-}")
legend()
xlabel(L"time")
ylabel(L"\langle G(τ,k) \rangle")
savefig(abspath(@__DIR__, string("../plots/G_plusMinus_k_",lat.Lm,"_",lat.Ln,"_",lat.Nt)))