

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))
equation = 35
folder = string("SublatticeSpinDifference_2_2_eq",equation)
conf_folder = "configurations"
subfolder = "Intermediate_results"
configurationfile = "run"
n_mass = 5
n_alpha = 12
file_path_config(i) = abspath(@__DIR__,string("../results/",folder,"/",conf_folder,"/",configurationfile,"_$i.csv"))
file_folder = string(folder,"/",subfolder)
lat, HMC_par = Read_Settings(file_path_config(1), ["hmc", "lat"])


Δn_array = zeros((n_mass, n_alpha, 4))   # The for things which are stored are: {Δn, err_Δn, ϵ, mass}

Filename(i,mass) = string(file_folder,"/SublatticeSpin_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
ms = [0.1,0.2,0.3,0.4,0.5]

for j = 1:length(ms)[1]
    for i = 1:n_alpha
        par = Read_Settings(file_path_config(i), ["par"])
        if isfile(abspath(@__DIR__,string("../results/",Filename(i,ms[j]),".txt")))
            res_Δn = ReadResult(Filename(i,ms[j]))    
            Δn_M = mean(res_Δn[HMC_par.offset:end])
            Δn2_M = mean(res_Δn[HMC_par.offset:end].^2)
            # err_Δn_M = std(res_Δn[offset:end])
            err_Δn_M = sqrt(abs(Δn2_M-Δn_M^2)/(HMC_par.Nsamples-HMC_par.offset-1))
            Δn_array[j, i, 1] = Δn_M
            Δn_array[j, i, 2] = err_Δn_M
        end
        Δn_array[j, i, 3] = par.ϵ
        Δn_array[j, i, 4] = ms[j]        
    end
    CreateFigure((300/137) ./Δn_array[j,:,3], Δn_array[j,:,1], folder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[j]*10)), 
                y_err = Δn_array[j,:,2] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle \Delta n \rangle$", 
                figure_title=string(L"$\Delta n$ for m = ",ms[j]))
end

αs = (300/137) ./Δn_array[1,:,3]

 
clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
title(L"$\Delta n$ for varying interactiong strength")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle Δn \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_all_masses.png")))

m_x = 0.0:0.01:0.6

m_0 = zeros(Float64,size(αs)[1])
clf()
for j = 1:size(αs)[1]
    fit = curve_fit(Polynomial, ms, Δn_array[:,j,1], 2)
    y0b = fit.(m_x) 
    m_0[j] = fit.(0.0)
end

clf()
plot(αs, m_0, "o")
xlabel(L"$\alpha_eff$")
ylabel(L"$\langle \Delta n \rangle$")
title("extrapolated E[Δn] for $lat")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_mass_extrapolation.png")))


