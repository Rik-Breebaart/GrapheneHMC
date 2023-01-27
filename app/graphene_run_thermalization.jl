#!/bin/julia
#=This file will contain the computations of the greens function of the non-interacting graphene tight binding model.

The correlator will be computed in two ways. using the Fermionic matrix and using the analytical summation. 
We will also check both in k-space and position space keeping the temporal component the same.
=#

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianNoInteractions.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))

runlabel = ARGS[1]
folder = ARGS[2]
filepath = abspath(@__DIR__,string("../results/",folder,"/configurations/",ARGS[3]))
subfolder = string(folder,"/Intermediate_results")
Thermalization_folder = string(folder,"/thermalization")

rng = MersenneTwister()
lat, par, HMC_par = Read_Settings(filepath, ["par", "lat", "hmc"])
File_phi = string(subfolder,"/Phi_interacting_eq",HMC_par.equation,"_Nt_",lat.Nt,"_runlabel_",runlabel)

ϕ_init = ones(ComplexF64,lat.D).*100.0

V = partialScreenedCoulomb_potential(par, lat)
if HMC_par.equation == 41
    M_part = FermionicMatrix_int_41_saved_part(par, lat)
elseif HMC_par.equation == 35
    M_part = FermionicMatrix_int_35_saved_part(V, par, lat)
end

function M_function(ϕ)
    if HMC_par.equation == 41
        FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)
    elseif HMC_par.equation == 35
        FermionicMatrix_int_35_phi_part(ϕ, M_part, par, lat)
    end
end

function ∇S_M(ϕ, χ)
    if HMC_par.equation == 41
        ∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
    elseif HMC_par.equation == 35
        ∇S_M_eq35_cg(ϕ, χ, M_function(ϕ), par, lat)        
    end
end

S(ϕ, χ) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)
∇S_V(ϕ, χ) = ∇S_V_cg(ϕ, V, par, lat)
D = lat.D

configurations, nreject = HybridMonteCarlo(S, ∇S_V, ∇S_M, M_function, D, HMC_par.path_length, HMC_par.step_size, HMC_par.m_sw, HMC_par.Nsamples; rng=rng, position_init=ϕ_init, print_time=true, print_accept=true, burn_in=HMC_par.burn_in, storefolder= File_phi)
println((HMC_par.Nsamples-nreject)/HMC_par.Nsamples, " percent for m=",par.mass, " and α=",(300/137)/par.ϵ)
res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in 1:HMC_par.Nsamples]

StoreResult(string(subfolder,"/SublatticeSpin_interacting_eq",HMC_par.equation,"_Nt_",lat.Nt,"_runlabel_",runlabel), res_Δn)
CreateFigure(res_Δn, subfolder,string("SublatticeSpin_interacting_eq",HMC_par.equation,"_Nt_",lat.Nt,"_runlabel_",runlabel) ,x_label="sample", y_label=L"$\Delta n$", fmt="-", figure_title=string(L"$\Delta n$ for m = ",par.mass, L" and $\alpha_{eff}$ = ", (300/137)/par.ϵ))

res_ϕ_A = [HubbardStratonovichField(configurations[i,:], 0, par, lat) for i in 1:HMC_par.Nsamples]
CreateFigure(res_ϕ_A, Thermalization_folder, string("HubbardFieldA_eq",HMC_par.equation,"_Nt_",lat.Nt,"_runlabel_",runlabel) ,fmt="-", 
            x_label="sample", y_label=L"$\langle ϕ_A \rangle$", figure_title="Thermalization of hubbard field on A sublattice")

Δn_M = mean(res_Δn[HMC_par.offset:end])
Δn2_M = mean(res_Δn[HMC_par.offset:end].^2)

err_Δn_M = sqrt(abs(Δn2_M-Δn_M^2)/(HMC_par.Nsamples-HMC_par.offset-1))
@show Δn_M
@show err_Δn_M
