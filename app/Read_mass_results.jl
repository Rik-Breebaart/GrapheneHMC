

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))
equation = 41
folder = string("SublatticeSpinDifference_4_4_Nt_16_eq",equation,"_final")
conf_folder = "configurations"
subfolder = "Intermediate_results"
configurationfile = "run"
extrapolate_folder = string(folder,"/extrapolate")
n_mass = 5
n_alpha = 10
file_path_config(i) = abspath(@__DIR__,string("../results/",folder,"/",conf_folder,"/",configurationfile,"_$i.csv"))
file_folder = string(folder,"/",subfolder)
lat, HMC_par = Read_Settings(file_path_config(1), ["hmc", "lat"])


Δn_array = zeros((n_mass, n_alpha, 5))   # The for things which are stored are: {Δn, err_Δn, ϵ, mass, β}
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
        Δn_array[j, i, 5] = par.β        
    end
    CreateFigure((300/137) ./Δn_array[j,:,3], Δn_array[j,:,1], folder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[j]*10)), 
                y_err = Δn_array[j,:,2] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle \Delta n \rangle$", 
                figure_title=string(L"$\Delta n$ for m = ",ms[j],L", $\beta$ = ",round(Δn_array[j, 1, 5],digits=3)))
end

αs = (300/137) ./Δn_array[1,:,3]

 
clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
ylim([0.0, maximum(ms)])
grid()
title(string(L"$\Delta n$ for varying interactiong strength, $\beta$ = ",round(Δn_array[1, 1, 5],digits=3)))
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle Δn \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_all_masses.png")))

m_x = 0.0:0.01:0.6

int_ms = [1,2,3,4,5]
plot_extrapolate = true
m_0 = zeros(Float64,size(αs)[1])
for i = 1:size(αs)[1]
    fit = curve_fit(Polynomial, ms[int_ms], Δn_array[int_ms,i,1], 2)
    y0b = fit.(m_x) 
    m_0[i] = fit.(0.0)
    if plot_extrapolate
        clf()
        errorbar(ms,
            Δn_array[:, i, 1],
            yerr= Δn_array[:, i, 2],
            fmt="o") 
        plot(m_x, y0b, "--", linewidth=1)
        xlabel("mass")
        ylabel(L"\langle \Delta n \rangle")
        title(string(L"Mass extrapolation for $\alpha = $",round(αs[i], digits=3), L" $\Delta n(0) $= ",round(fit.(0.0), digits=3)))
        savefig(abspath(@__DIR__,string("../results/",extrapolate_folder,"/SublatticeSpin_interacting_eq",HMC_par.equation,"_mass_extrapolation_runlabel_",i,".png")))
    end 
end

clf()
plot(αs, m_0, "o")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle \Delta n \rangle$")
title(string(L"extrapolated $\langle \Delta n\rangle$ for ",lat,L"$\beta$ = ",round(Δn_array[1, 1, 5],digits=3)))
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_mass_extrapolation.png")))
