

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit, Test
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianInteractions.jl"))
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__, "../src/interactions.jl"))
include(abspath(@__DIR__, "../src/actionComponents.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))
equation = 41
folder = string("SublatticeSpinDifference_6_6_Nt_16_eq",equation,"_final")
conf_folder = "configurations"
subfolder = "Intermediate_results"
configurationfile = "run"
extrapolate_folder = string(folder,"/extrapolate")
n_mass = 5
n_alpha = 10
file_path_config(i) = abspath(@__DIR__,string("../results/",folder,"/",conf_folder,"/",configurationfile,"_$i.csv"))
file_folder = string(folder,"/",subfolder)
lat, HMC_par = Read_Settings(file_path_config(1), ["hmc", "lat"])

meas_array = zeros((n_mass, n_alpha, 9))   # The for things which are stored are: {Δn, err_Δn, ϵ, mass, β}
Filename(i,mass) = string(file_folder,"/SublatticeSpin_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
File_phi(i, mass) = string(file_folder,"/Phi_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
ms = [0.1, 0.2, 0.3, 0.4, 0.5]
for j = 1:length(ms)[1]
    for i = 1:n_alpha
        par = Read_Settings(file_path_config(i), ["par"])

        if isfile(abspath(@__DIR__,string("../results/",Filename(i,ms[j]),".txt")))
            #create the M_function to be used for the computation of the observables.
            change_mass(ms[j], par)
            V = partialScreenedCoulomb_potential(par, lat)
            if equation == 41
                M_part = FermionicMatrix_int_41_saved_part(par, lat)
            elseif equation == 35
                M_part = FermionicMatrix_int_35_saved_part(V, par, lat)
            end
        
            function M_function(ϕ)
                if equation == 41
                    FermionicMatrix_int_41_phi_part(ϕ, M_part, par, lat)
                elseif equation == 35
                    FermionicMatrix_int_35_phi_part(ϕ, M_part, par, lat)
                end
            end
            configurations = ReadResult(File_phi(i,ms[j]), complex=false)
            meas = [S_33_min_min(-M_function(configurations[i,:]), par, lat) for i in 1:HMC_par.Nsamples] 
            meas_S3_min = getindex.(meas, 1)
            meas_S33_min_min = getindex.(meas, 2)
            meas_S3_plus = getindex.(meas, 3)


            StoreResult(string(file_folder,"/S33_min_min_interacting_eq",equation,"_m_",floor(Integer,ms[j]*100),"_runlabel_",i), meas_S33_min_min)
            StoreResult(string(file_folder,"/S3_plus_interacting_eq",equation,"_m_",floor(Integer,ms[j]*100),"_runlabel_",i), meas_S3_plus)
            StoreResult(string(file_folder,"/S3_min_interacting_eq",equation,"_m_",floor(Integer,ms[j]*100),"_runlabel_",i), meas_S3_min)
            
            S3_min = mean(meas_S3_min[HMC_par.offset:end])
            S3_min2 = mean(meas_S3_min[HMC_par.offset:end].^2)
            S3_plus = mean(meas_S3_plus[HMC_par.offset:end])
            S3_plus2 = mean(meas_S3_plus[HMC_par.offset:end].^2)
            S33_M = mean(meas_S33_min_min[HMC_par.offset:end])
            S33_2_M = mean(meas_S33_min_min[HMC_par.offset:end].^2)
            err_S3_min_M = sqrt(abs(S3_min2-S3_min^2)/(HMC_par.Nsamples-HMC_par.offset-1))
            err_S33_M = sqrt(abs(S33_2_M-S33_M^2)/(HMC_par.Nsamples-HMC_par.offset-1))
            err_S3_plus = sqrt(abs(S3_plus2-S3_plus^2)/(HMC_par.Nsamples-HMC_par.offset-1))
            meas_array[j, i, 1] = S3_min
            meas_array[j, i, 2] = err_S3_min_M
            meas_array[j, i, 3] = S33_M
            meas_array[j, i, 4] = err_S33_M
            meas_array[j, i, 5] = S3_plus
            meas_array[j, i, 6] = err_S3_plus
        end
        meas_array[j, i, 7] = par.ϵ
        meas_array[j, i, 8] = ms[j]   
        meas_array[j, i, 9] = par.β 

    end
    CreateFigure((300/137) ./meas_array[j,:,7], meas_array[j,:,1], folder,string("SublatticeSpin_interacting_eq41_m_",floor(Integer,ms[j]*10)), 
    y_err = meas_array[j,:,2] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle \Delta n \rangle$", 
    figure_title=string(L"$\Delta n$ for m = ",ms[j],L", $\beta$ = ",round(meas_array[j, 1, 9],digits=3)))

    CreateFigure((300/137) ./meas_array[j,:,7], meas_array[j,:,3], folder,string("S33_min_min_interacting_eq41_m_",floor(Integer,ms[j]*10)), 
    y_err = meas_array[j,:,4] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle S^{33}_{--} \rangle$", 
    figure_title=string(L"$S^{33}_{--}$ for m = ",ms[j],L", $\beta$ = ",round(meas_array[j, 1, 9],digits=3)))

    CreateFigure((300/137) ./meas_array[j,:,7], meas_array[j,:,5], folder,string("S3_plus_interacting_eq41_m_",floor(Integer,ms[j]*10)), 
    y_err = meas_array[j,:,6] ,x_label=L"$\alpha_{eff}$", y_label=L"$\langle S^{33}_{--} \rangle$", 
    figure_title=string(L"$S^{3}_{+}$ for m = ",ms[j],L", $\beta$ = ",round(meas_array[j, 1, 9],digits=3)))
end


αs = (300/137) ./meas_array[1,:,7]

clf()
for i in 1:length(ms)[1]
errorbar(αs, meas_array[i,:,1], yerr=meas_array[i,:,2],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
ylim([0.0, maximum(ms)])
grid()
title(string(L"$S3_min$ for varying interactiong strength, $\beta$ = ",round(meas_array[1, 1, 9],digits=3)))
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle meas \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/SublatticeSpin_interacting_eq41_all_masses.png")))

clf()
for i in 1:length(ms)[1]
errorbar(αs, meas_array[i,:,3], yerr=meas_array[i,:,4],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
# ylim([0.0, maximum(ms)])
grid()
title(string(L"$S^{33}_{--}$ for varying interactiong strength, $\beta$ = ",round(meas_array[1, 1, 9],digits=3)))
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{33}_{--} \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/S33_min_min_interacting_eq41_all_masses.png")))

clf()
for i in 1:length(ms)[1]
errorbar(αs, meas_array[i,:,5], yerr=meas_array[i,:,6],fmt="o", label=string("m = ",ms[i]))
end 
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
# ylim([0.0, maximum(ms)])
grid()
title(string(L"$S^{3}_{+}$ for varying interactiong strength, $\beta$ = ",round(meas_array[1, 1, 9],digits=3)))
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{+} \rangle$")
savefig(abspath(@__DIR__,string("../results/",folder,"/S3_plus_interacting_eq41_all_masses.png")))
