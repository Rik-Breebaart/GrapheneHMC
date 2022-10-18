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
extrapolate_folder = string(folder,"/extrapolate")

rng = MersenneTwister()
lat, par, HMC_par = Read_Settings(filepath, ["par", "lat", "hmc"])
ms = [0.5, 0.4, 0.3, 0.2, 0.1]
ϕ_init = zeros(ComplexF64,lat.D)
storage = zeros( (length(ms)[1],2))
for i in 1:length(ms)[1]
    global ϕ_init
    if i==1
        ϕ_init = rand(rng,(lat.D)).*5.0
    end
    change_mass(ms[i], par)

    #we will look at equation 41
    V = coulomb_potential(par, lat)
    M_part = FermionicMatrix_int_41_saved_part(par, lat)
    M_function(ϕ) = FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)

    S(ϕ, χ) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)
    ∇S_V(ϕ, χ) = ∇S_V_cg(ϕ, V, par, lat)
    ∇S_M(ϕ, χ) = ∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
    D = lat.D

    configurations, nreject = HybridMonteCarlo(S, ∇S_V, ∇S_M, M_function, D, HMC_par.path_length, HMC_par.step_size, HMC_par.m_sw, HMC_par.Nsamples; rng=rng, position_init=ϕ_init, print_time=true, print_accept=true, burn_in=HMC_par.burn_in)
    ϕ_init = configurations[end,:]
    println((HMC_par.Nsamples-nreject)/HMC_par.Nsamples, " percent for m=",par.mass, " and α=",(300/137)/par.ϵ)
    res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in 1:HMC_par.Nsamples]
    StoreResult(string(subfolder,"/SublatticeSpin_interacting_eq41_m_",floor(Integer,par.mass*10),"_runlabel_",runlabel), res_Δn)
    CreateFigure(res_Δn, subfolder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,par.mass*10),"_runlabel_",runlabel) ,x_label="sample", y_label=L"$\Delta n$", fmt="-", figure_title=string(L"$\Delta n$ for m = ",par.mass, L" and $\alpha_{eff}$ = ", (300/137)/par.ϵ))

    Δn_M = mean(res_Δn[HMC_par.offset:end])
    Δn2_M = mean(res_Δn[HMC_par.offset:end].^2)

    err_Δn_M = sqrt(abs(Δn2_M-Δn_M^2)/(HMC_par.Nsamples-HMC_par.offset-1))
    storage[i,1] = Δn_M 
    storage[i,2] = err_Δn_M
    @show Δn_M
    @show err_Δn_M
end 

m_x = 0.0:0.01:0.6

clf()
errorbar(ms,
    storage[:,1], 
    yerr= storage[:,2],
    fmt="o") 
fit = curve_fit(Polynomial, ms, storage[:,1], 2)
y0b = fit.(m_x) 
plot(m_x, y0b, "--", linewidth=1)
xlabel("mass")
ylabel(L"\langle \Delta n \rangle")
title(string(L"Mass extrapolation for $\alpha = $",round((300/137)./par.ϵ, digits=3), L" $\Delta n(0) $= ",round(fit.(0.0), digits=3)))
savefig(abspath(@__DIR__,string("../results/",extrapolate_folder,"/SublatticeSpin_interacting_eq41_mass_extrapolation_runlabel_",runlabel,".png")))



