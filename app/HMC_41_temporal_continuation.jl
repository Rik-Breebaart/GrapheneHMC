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
par = Parameters(2.0, 0.5, 2.2/α, 0.5)
Nts = [8,12,16,20,24]
Δn_array = zeros((length(Nts)[1],2))
lat_0 = Lattice(6, 6, 1)
for i = 1:length(Nts)[1]
    lat = Lattice(lat_0.Lm, lat_0.Ln, Nts[i])
    #we will look at equation 41
    V = partialScreenedCoulomb_potential(par, lat)
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
    configurations, nreject = HybridMonteCarlo(S::Function, ∇S::Function, M_function::Function, D::Integer, path_length, step_size, Nsamples::Integer; rng=rng, position_init=10.0, print_H=true)
    @show (Nsamples-nreject)/Nsamples

    res_Δn = [Δn(M_function(configurations[i,:]), par, lat) for i in 1:Nsamples]
    clf()
    plot(res_Δn)
    xlabel("sweeps")
    ylabel(L"$\Delta n$")
    savefig(abspath(@__DIR__,string("../plots/SublatticeSpin_hmc_thermilization_41_continues_",lat.Lm,"_",lat.Ln,"_",lat.Nt)))
    Δn_array[i,1] = mean(res_Δn)
    Δn_array[i,2] = std(res_Δn)
end

clf()
Nts_x = -0.05:0.01:((1/8)+0.05)

errorbar(1 ./Nts, Δn_array[:,1],yerr=Δn_array[:,2],fmt="o")
fit = curve_fit(Polynomial, 1 ./Nts, Δn_array[:,1], 1)
y0b = fit.(Nts_x) 
plot(Nts_x, y0b, "--", linewidth=1)
ylabel("1/Nt")
xlabel(L"$\langle \Delta n \rangle")
xlim([-0.05, (1/8)+0.05])
grid()
savefig(abspath(@__DIR__,string("../plots/SublatticeSpin_hmc_thermilization_41_continues_",lat_0.Lm,"_",lat_0.Ln,".png")))
