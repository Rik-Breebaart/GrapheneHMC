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

#set configuration settings
lat = Lattice(4, 4, 12)
ms = [0.5, 0.4, 0.3, 0.2, 0.1]
ϵs = LinRange(0.45, 2.5, 21)
αs = (300/137)./ϵs
rng = MersenneTwister(123)

path_length = 10.0
step_size = 1.0
m = 10  #sexton weingarten split Fermionic substeps
Nsamples= 500
burn_in = 100
offset = floor(Integer, Nsamples*0.3)

β = 2.0


Δn_array = zeros((length(ms)[1],length(ϵs)[1],2))
Δn_analytical = zeros((length(ms)[1]))
folder = storage_folder("SublatticeSpinDifference", "41", β, lat)
sub_folder = storage_folder(string(folder,"/storage_intermediate"),"41",β,lat)
ϕ_init = zeros(ComplexF64,lat.D)
for i in 1:length(ms)[1]
    for j in 1:length(ϵs)[1]
        global ϕ_init
        if i==1 && j==1
            ϕ_init = rand(rng,(lat.D)).*10.0
        end

        if isapprox(1/ϵs[j],0.0,atol=0.0001)
            par = Parameters(β, ms[i], 10000.0, 0.5)
            M_no_int = FermionicMatrix_no_int(par, lat)
            Δn_analytical[i] = real.(Δn_no_int(par, lat))
            Δn_array[i,j,1] = Δn(M_no_int, par, lat)
        else 
            par = Parameters(β, ms[i], ϵs[j], 0.5)
            #we will look at equation 35
            V = coulomb_potential(par, lat)
            M_part = FermionicMatrix_int_41_saved_part(par, lat)
            M_function(ϕ) = FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)
            # M_function(ϕ) = FermionicMatrix_int_41(ϕ, V, par, lat)
            S(ϕ, χ) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)
            ∇S_V(ϕ, χ) = ∇S_V_cg(ϕ, V, par, lat)
            ∇S_M(ϕ, χ) = ∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
            D = lat.D

            configurations, nreject = HybridMonteCarlo(S, ∇S_V, ∇S_M, M_function, D, path_length, step_size, m, Nsamples; rng=rng, position_init=ϕ_init, print_time=true, print_accept=true, burn_in=burn_in)
            ϕ_init = configurations[end,:]
            println((Nsamples-nreject)/Nsamples, " percent for m=",ms[i], " and α=",(300/137)/ϵs[j])
            res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in 1:Nsamples]
            StoreResult(string(sub_folder,"/SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[i]*10),"_alpha_",floor(Int,((300/137)/ϵs[j])*10)), res_Δn)
            CreateFigure(res_Δn, sub_folder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[i]*10),"_alpha_",floor(Int,((300/137)/ϵs[j])*10)),x_label="sample", y_label=L"$\Delta n$", fmt="-", figure_title=string(L"$\Delta n$ for m = ",ms[i], L" and $\alpha_{eff}$ = ", (300/137)/ϵs[j]))
            
            Δn_M = mean(res_Δn[offset:end])
            Δn2_M = mean(res_Δn[offset:end].^2)
            # err_Δn_M = std(res_Δn[offset:end])
            err_Δn_M = sqrt(Δn2_M-Δn_M^2/(Nsamples-offset-1))
            Δn_array[i,j,1] = Δn_M
            Δn_array[i,j,2] = err_Δn_M
        end 
    end
    StoreResult(string(folder,"/SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[i]*10)), Δn_array[i,:,:])
    CreateFigure(αs, Δn_array[i,:,1], folder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[i]*10)), 
                y_err = Δn_array[i,:,2] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle \Delta n \rangle$", 
                figure_title=string(L"$\Delta n$ for m = ",ms[i]))
end

 
clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
title(L"$\Delta n$ for varying interactiong strength")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle Δn \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41.png")))

clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
title(L"$\Delta n$ for varying interactiong strength")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle Δn \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41.png")))

m_x = 0.0:0.01:0.6

m_0 = zeros(Float64,size(αs)[1])
clf()
for j = 1:size(ϵs)[1]
    errorbar(ms,
        Δn_array[:,j,1], 
        yerr=Δn_array[:,j,2],
        fmt="o") 
    fit = curve_fit(Polynomial, ms, Δn_array[:,j,1], 2)
    y0b = fit.(m_x) 
    m_0[j] = fit.(0.0)
    plot(m_x, y0b, "--", linewidth=1)
end
xlabel("mass")
ylabel(L"\langle Δn \rangle")
title("$lat")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_mass_clutter.png")))

clf()
plot(αs, m_0, "o")
xlabel(L"\alpha")
ylabel(L"\langle Δn \rangle")
title("extrapolated E[Δn] for $lat")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_mass_extrapolation.png")))

