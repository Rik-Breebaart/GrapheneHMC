#= 
In this file I will test the HMC implementation for graphene looking at the correlator as an observable.

=#

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))


α = 1.87
mass = 0.5
par = Parameters(2.0, mass, (300/137)/α, 0.5)
path_length = 10.0
step_size = 0.5
m = 5
Nsamples= 10000
burn_in = 100
offset = floor(Integer, 0.2*Nsamples)

Nts = [8, 10, 12, 16]
Δn_array = zeros((length(Nts)[1],2))
lat = Lattice(6, 6, 1)

folder = storage_folder("Themporal_Continuem", "41", par, lat)

sub_folder = storage_folder(string(folder,"/storage_intermediate"),"41",par,lat)

for i = 1:length(Nts)[1]
    change_lat(lat, Nt = Nts[i])
    #we will look at equation 41
    V = partialScreenedCoulomb_potential(par, lat)
    M_part = FermionicMatrix_int_41_saved_part(par, lat)
    M_function(ϕ) = FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)
    # M_function(ϕ) = FermionicMatrix_int_41(ϕ, par, lat)

    rng = MersenneTwister(123)
    S(ϕ, χ) = Action_V_cg(ϕ, V, par ,lat) + Action_M_cg(χ, M_function(ϕ), par ,lat)
    ∇S_V(ϕ, χ) = ∇S_V_cg(ϕ, V, par, lat)
    ∇S_M(ϕ, χ) = ∇S_M_eq41_cg(ϕ, χ, M_function(ϕ), par, lat)
    D = lat.D
    ϕ_init = ones(lat.D)*100

    configurations, nreject = HybridMonteCarlo(S, ∇S_V, ∇S_M, M_function, D, path_length, step_size, m, 
                                                Nsamples; rng=rng, position_init=ϕ_init, print_H=true, burn_in=burn_in)
    @show (Nsamples-nreject)/Nsamples

    res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in 1:Nsamples]

    
    Δn_M = mean(res_Δn[offset:end])
    Δn2_M = mean(res_Δn[offset:end].^2)
    # err_Δn_M = std(res_Δn[offset:end])
    err_Δn_M = sqrt(abs(Δn2_M-Δn_M^2)/(Nsamples-offset-1))
    Δn_array[i,1] = Δn_M
    Δn_array[i,2] = err_Δn_M
    StoreResult(string(sub_folder,"/SublatticeSpin_interacting_eq41_m_",floor(Integer,par.mass*10),"_alpha_",floor(Int,((300/137)/par.ϵ)*10),"_Nt_",lat.Nt), res_Δn)
    CreateFigure(res_Δn, sub_folder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,par.mass*10),"_alpha_",floor(Int,((300/137)/par.ϵ)*10),"_Nt_",lat.Nt),
                x_label="sample", y_label=L"$\Delta n$", fmt="-", figure_title=string(L"$\beta$ = ", par.β ,L"$\alpha $= ",(300/137)/par.ϵ, " m = ", par.mass))
    
    res_ϕ_A = [HubbardStratonovichField(configurations[i,:], 0, par, lat) for i in 1:Nsamples]
    CreateFigure(res_ϕ_A, sub_folder,string("HubbardFieldA_eq41_Nt_",lat.Nt),fmt="-", 
                x_label="sample", y_label=L"$\langle ϕ_A \rangle$", figure_title="Thermalization of hubbard field on A sublattice")
end

clf()
Nts_x = 0.0:0.01:((1/8)+0.05)

errorbar(1 ./Nts, Δn_array[:,1],yerr=Δn_array[:,2],fmt="o")
fit = curve_fit(Polynomial, 1 ./Nts, Δn_array[:,1], 1)
y0b = fit.(Nts_x) 
plot(Nts_x, y0b, "--", linewidth=1,)
xlabel("1/Nt")
ylabel(L"$\langle \Delta n \rangle$")
title(string(L"$\beta$ = ", par.β ,L"$\alpha $= ",round((300/137)/par.ϵ,digits=2), " m = ", par.mass))
xlim([0.0, (1/8)+0.05])
grid()
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_hmc_thermilization_41_continues_mass_",floor(Integer,par.mass*10),"_",lat.Lm,"_",lat.Ln,".png")))
