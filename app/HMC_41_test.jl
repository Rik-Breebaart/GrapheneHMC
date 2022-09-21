#= 
In this file I will test the HMC implementation for graphene looking at the correlator as an observable.

=#

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))



lat = Lattice(4, 4, 24)
lat_analytic = Lattice(lat.Lm, lat.Ln, 24)
par = Parameters(2.0, 0.5, 10.0, 0.5)

particle_x = Particle(1, 1, 0, 1)
particle_y = Particle(1, 1, 0, 1)

#analytical correlator without interactions both in real and k-space
correlator = greensFunctionGraphene_spatial(particle_x,particle_y, par, lat_analytic)
k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
k_b = (4*pi)/(3*lat.a)* [0,1]
ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b
correlator_momentum = greensFunctionGraphene_kspace(ks(1,1), par, lat_analytic)
Δn_analytical = Δn_no_int(par, lat)

#we will look at equation 41
V = coulomb_potential(par, lat)
M_part = FermionicMatrix_int_41_saved_part(par, lat)
M_function(ϕ) = FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)
# M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)

rng = MersenneTwister(123)
S(ϕ, χ) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)
∇S(ϕ, χ) = ∇S_V_cg(ϕ, V, par, lat)+∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
D = lat.D
path_length = 10.0
step_size = 0.05
Nsamples= 100

configurations, nreject = HybridMonteCarlo(S::Function, ∇S::Function, M_function::Function, D::Integer, path_length, step_size, Nsamples::Integer; rng=rng, position_init=100 .*ones(lat.D), print_H=true))
@show (Nsamples-nreject)/Nsamples

res_spatial = [greens_function_spatial(M_function(configurations[i,:]), particle_x, particle_y, par, lat) for i in 1:Nsamples]
res_momentum = [greens_function_kspace(M_function(configurations[i,:]), ks(1,1), par, lat) for i in 1:Nsamples]
res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in 1:Nsamples]
clf()
plot(res_Δn)
xlabel("sweeps")
ylabel(L"$\Delta n$")
savefig(abspath(@__DIR__,"../plots/SublatticeSpin_hmc_thermilization_41"))

res_ϕ_A = [HubbardStratonovichField(configurations[i,:],0, par, lat) for i in 1:Nsamples]
clf()
plot(res_ϕ_A)
xlabel("sweeps")
ylabel(L"$\langle \phi_A \rangle$")
savefig(abspath(@__DIR__,"../plots/phi_A_hmc_thermilization_41"))
Δn_M = mean(res_Δn)
correlator_M = mean(res_spatial)
correlator_M_momentum = mean(res_momentum)

name = ["A","B"]
@show Δn_analytical, Δn_M

τ = (0:1:lat.Nt-1).*(par.β/lat.Nt)
τ_analytic = (0:1:lat_analytic.Nt-1).*(par.β/lat_analytic.Nt)
for i=[1,2]
    for j = [1,2]
        clf()
        if i==j
            semilogy(τ_analytic, real.(correlator[:,i,j]), "*", label="analytical")
            semilogy(τ, real.(correlator_M[:,i,j]),".", label="Fermionic Matrix")
        else 
            plot(τ_analytic, real.(correlator[:,i,j]), "*", label="analytical")
            plot(τ, real.(correlator_M[:,i,j]),".", label="Fermionic Matrix")
        end
        legend()
        xlabel(L"time")
        ylabel(L"\langle G(τ,x,y) \rangle")
        savefig(abspath(@__DIR__, string("../plots/HMC_41_G",name[i],name[j])))
    end 
end



for i=[1,2]
    for j = [1,2]
        clf()
        if i==j
            semilogy(τ_analytic, real.(correlator_momentum[:,i,j]), "*", label="analytical")
            semilogy(τ, real.(correlator_M_momentum[:,i,j]), ".", label="Fermionic")
        else 
            plot(τ_analytic, real.(correlator_momentum[:,i,j]), "*", label="analytical")
            plot(τ, real.(correlator_M_momentum[:,i,j]), ".", label="Fermionic")
        end
        legend()
        xlabel(L"time")
        ylabel(L"\langle G(τ,k) \rangle")
        savefig(abspath(@__DIR__, string("../plots/HMC_41_G_k",name[i],name[j])))
    end 
end