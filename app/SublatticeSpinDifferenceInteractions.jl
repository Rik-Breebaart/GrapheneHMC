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
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))


lat = Lattice(4, 4, 16)
ms = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ϵs = LinRange(0.45, 1.0, 11)
rng = MersenneTwister(123)

Δn_array = zeros((length(ms)[1],length(ϵs)[1],2))
Δn_analytical = zeros((length(ms)[1]))
par_0 = Parameters(2.0, 0.0, 1.0, 0.5)
for j in 1:length(ϵs)[1]
    for i in 1:length(ms)[1]
        if isapprox(1/ϵs[j],0.0,atol=0.0001)
            par = Parameters(par_0.β, ms[i], 10000.0, 0.5)
            M_no_int = FermionicMatrix_no_int(par, lat)
            Δn_analytical[i] = real.(Δn_no_int(par, lat))
            Δn_array[i,j,1] = Δn(M_no_int, par, lat)
        else 
            par = Parameters(par_0.β, ms[i], ϵs[j], 0.5)
            #we will look at equation 35
            V = coulomb_potential(par, lat)
            M_part = FermionicMatrix_int_41_saved_part(par, lat)
            M_function(ϕ) = FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)
            # M_function(ϕ) = FermionicMatrix_int_41(ϕ, V, par, lat)
            S(ϕ, χ) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)
            ∇S(ϕ, χ) = ∇S_V_cg(ϕ, V, par, lat)+∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
            D = lat.D
            path_length = 10.0
            step_size = 0.05
            Nsamples= 100
            offset = 10
            configurations, nreject = HybridMonteCarlo(S::Function, ∇S::Function, M_function::Function, D::Integer, path_length, step_size, Nsamples::Integer; rng=rng, position_init=10.0)
            println((Nsamples-nreject)/Nsamples, "percent for $par")
            res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in offset:Nsamples]
            clf()
            plot(res_Δn)
            xlabel("sample")
            ylabel(L"\Delta n")
            savefig(abspath(@__DIR__,string("../plots/SublatticeSpin_interacting_m_",i,"_alpha_",j,".png")))
            Δn_M = mean(res_Δn)
            err_Δn_M = std(res_Δn)
            Δn_array[i,j,1] = Δn_M
            Δn_array[i,j,2] = err_Δn_M
        end 
    end
end

clf()
for i in 1:length(ms)[1]
    errorbar((300/137)./ϵs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
    # plot(αs, Δn_analytical[i].*ones(length(αs)[1]), label=string("no_int m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
title(L"$\Delta n$ for varying interactiong strength")
xlabel("m")
ylabel("Δn")
savefig(abspath(@__DIR__,"../plots/SublatticeSpin_interacting.png"))
