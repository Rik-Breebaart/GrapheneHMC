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

folder = [string("SublatticeSpinDifference_4_4_eq",equation,"_2"),string("SublatticeSpinDifference_4_4_eq",equation), string("SublatticeSpinDifference_4_4_eq",equation,"_4")] # string("SublatticeSpinDifference_4_4_eq",equation,"_3"),
# folder = [string("SublatticeSpinDifference_4_4_eq",equation,"beta_smaller_1")]
figure_storage = "plots"
conf_folder = "configurations"
subfolder = "Intermediate_results"
configurationfile = "run"
extrapolate_folder = string(folder,"/extrapolate")
n_mass = 5
n_alphas = [5,12,7]
n_alpha_start = [1,2,1]
file_path_config(i,f) = abspath(@__DIR__,string("../results/",folder[f],"/",conf_folder,"/",configurationfile,"_$i.csv"))
file_folder_fun(f) = string(folder[f],"/",subfolder)
lat, HMC_par = Read_Settings(file_path_config(1,1), ["hmc", "lat"])

Δn_array = zeros((n_mass, sum(n_alphas), 4))   # The for things which are stored are: {Δn, err_Δn, ϵ, mass}

Filename(i,mass,f) = string(file_folder_fun(f),"/SublatticeSpin_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
ms = [0.1, 0.2, 0.3, 0.4, 0.5]

intervalstep_tau = 1


for j = 1:length(ms)[1]
    ind_0 = 0
    for f =1:length(folder)
        for i = n_alpha_start[f]:n_alphas[f]
            par = Read_Settings(file_path_config(i,f), ["par"])
            if isfile(abspath(@__DIR__,string("../results/",Filename(i,ms[j],f),".txt")))
                res_Δn = ReadResult(Filename(i,ms[j],f))    
                Δn_M = mean(res_Δn[HMC_par.offset:intervalstep_tau:end])
                Δn2_M = mean(res_Δn[HMC_par.offset:intervalstep_tau:end].^2)
                # err_Δn_M = std(res_Δn[offset:end])
                err_Δn_M = sqrt(abs(Δn2_M-Δn_M^2)/(floor((HMC_par.Nsamples-HMC_par.offset-1)/intervalstep_tau)))
                Δn_array[j, i+ind_0, 1] = Δn_M
                Δn_array[j, i+ind_0, 2] = err_Δn_M
            end
            Δn_array[j, i+ind_0, 3] = par.ϵ
            Δn_array[j, i+ind_0, 4] = ms[j]        
        end
        ind_0 = sum(n_alphas[1:f])
        CreateFigure((300/137) ./Δn_array[j,:,3], Δn_array[j,:,1], figure_storage ,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[j]*10)), 
                    y_err = Δn_array[j,:,2] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle S^3_{-} \rangle$", 
                    figure_title=string(L"$\langle S^3_{-} \rangle$ for m = ",ms[j]))
    end
end

αs = (300/137) ./Δn_array[1,:,3]

 
clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
ylim([0.0, maximum(ms)])
grid()
title(L"$S^3_{-}$ for varying interaction strength")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{-} \rangle$")
savefig(abspath(@__DIR__,string("../results/",figure_storage,"/SublatticeSpin_interacting_eq41_all_masses.png")))

m_x = 0.0:0.01:0.6

int_ms = [1,2,3,4,5]
m_0 = zeros(Float64,size(αs)[1])
for i = 1:size(αs)[1]
    fit = curve_fit(Polynomial, ms[int_ms], Δn_array[int_ms,i,1], 2)
    y0b = fit.(m_x) 
    m_0[i] = fit.(0.0)
end

clf()
plot(αs, m_0, "o")
xlabel(L"$\alpha_eff$")
ylabel(L"$\langle S^3_{-} \rangle$")
title(L"$\langle S^{3}_{-}\rangle(m\rightarrow 0) $ for $\beta= 2.0$ ")
savefig(abspath(@__DIR__,string("../results/",figure_storage,"/SublatticeSpin_interacting_eq41_mass_extrapolation.png")))

clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
plot(αs, m_0, "o", label = L"m \rightarrow 0")
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
ylim([-0.1, maximum(ms)])
grid()
title(L"$\langle S^3_{-} \rangle$ for varying interaction strength")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{-} \rangle$")
savefig(abspath(@__DIR__,string("../results/",figure_storage,"/SublatticeSpin_interacting_eq41_all_masses_with_extrapolation.png")))


clf()
for i in 1:length(ms)[1]
    errorbar(αs, Δn_array[i,:,1], yerr=Δn_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
ylim([0.0, maximum(ms)])
grid()
title(L"$\langle S^3_{-} \rangle$ for varying interaction strength")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{-} \rangle$")
savefig(abspath(@__DIR__,string("../results/",figure_storage,"/SublatticeSpin_interacting_eq41_all_masses.png")))
