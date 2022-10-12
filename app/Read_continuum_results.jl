

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))

folder = "Continuum_limit"
conf_folder = "configurations"
subfolder = "Intermediate_results"
configurationfile = "run"
n_Nt = 3
file_path_config(i) = abspath(@__DIR__,string("../results/",folder,"/",conf_folder,"/",configurationfile,"_$i.csv"))
file_folder = string(folder,"/",subfolder)
par, HMC_par = Read_Settings(file_path_config(1), ["hmc", "par"])

α = (300/137)/par.ϵ

Δn_array = zeros((n_Nt, 3))   # The for things which are stored are: {Δn, err_Δn, ϵ, mass}

Filename(i,Nt) = string(file_folder,"/SublatticeSpin_interacting_eq41_Nt_",Nt,"_runlabel_",i)

for i = 1:n_Nt
    lat =  Read_Settings(file_path_config(i), ["lat"])
    res_Δn = ReadResult(Filename(i,lat.Nt))    
    Δn_M = mean(res_Δn[HMC_par.offset:end])
    Δn2_M = mean(res_Δn[HMC_par.offset:end].^2)
    # err_Δn_M = std(res_Δn[offset:end])
    err_Δn_M = sqrt((Δn2_M-Δn_M^2)/(HMC_par.Nsamples-HMC_par.offset-1))
    Δn_array[i, 1] = Δn_M
    Δn_array[i, 2] = err_Δn_M
    Δn_array[i, 3] = lat.Nt
end
Nts_x = 0.0:0.01:((1/8)+0.05)
 
clf()
errorbar(1 ./Δn_array[:,3],Δn_array[:,1], yerr=Δn_array[:,2],fmt="o")
fit = curve_fit(Polynomial, 1 ./Δn_array[:,3], Δn_array[:,1], 1)
y0b = fit.(Nts_x) 
plot(Nts_x, y0b, "--", linewidth=1,)
title(string(L"$\Delta n$ dependent on temporal spacing, $\Delta n \rightarrow ",round(fit.(0.0),digits=2)))
xlabel(L"$1/Nt$")
ylabel(L"$\langle Δn \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/Continuum_extrapolation.png")))



