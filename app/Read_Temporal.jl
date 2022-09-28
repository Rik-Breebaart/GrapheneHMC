
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
par = Parameters(2.0, 0.3, (300/137)/α, 0.5)
path_length = 10.0
step_size = 0.5
m = 10
Nsamples= 1000
burn_in = 100
offset = floor(Integer, 0.5*Nsamples)

Nts = [8,12,16,20,24]
Δn_array = zeros((length(Nts)[1],2))
lat = Lattice(2, 2, 1)

folder = "Themporal_Continuem_eq41_beta_20Lm_2Ln_2Nt_1_4"
sub_folder = "storage_intermediate_eq41_beta_20Lm_2Ln_2Nt_1_0"

Filename(i) = string(folder,"/",sub_folder,"/SublatticeSpin_interacting_eq41_m_3_alpha_18_Nt_",i)


for i = 1:length(Nts)[1]
    change_lat(lat, Nt = Nts[i])

    res_Δn = ReadResult(Filename(Nts[i]))    
    Δn_M = mean(res_Δn[offset:end])
    Δn2_M = mean(res_Δn[offset:end].^2)
    # err_Δn_M = std(res_Δn[offset:end])
    err_Δn_M = sqrt(Δn2_M-Δn_M^2/(Nsamples-offset-1))
    Δn_array[i,1] = Δn_M
    Δn_array[i,2] = err_Δn_M
end

clf()
Nts_x = 0.0:0.01:((1/8)+0.05)

errorbar(1 ./Nts, Δn_array[:,1],yerr=Δn_array[:,2],fmt="o")
fit = curve_fit(Polynomial, 1 ./Nts, Δn_array[:,1], 1)
y0b = fit.(Nts_x) 
plot(Nts_x, y0b, "--", linewidth=1,)
xlabel("1/Nt")
ylabel(L"$\langle \Delta n \rangle$")
title(string(L"$\beta$ = ", par.β ,L"$\alpha $= ",(300/137)/par.ϵ, " m = ", par.mass))
xlim([0.0, (1/8)+0.05])
grid()
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_hmc_thermilization_41_continues_mass_",floor(Integer,par.mass*10),"_",lat.Lm,"_",lat.Ln,".png")))
