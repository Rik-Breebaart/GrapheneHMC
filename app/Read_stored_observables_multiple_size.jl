

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
folder(L) = string("SublatticeSpinDifference_",L,"_",L,"_Nt_16_eq",equation,"_final")
conf_folder = "configurations"
subfolder = "Intermediate_results"
configurationfile = "run"

file_path_config(L,i) = abspath(@__DIR__,string("../results/",folder(L),"/",conf_folder,"/",configurationfile,"_$i.csv"))
file_folder(L) = string(folder(L),"/",subfolder)
extrapolate_folder(L) = string(folder(L),"/extrapolate")
Ls = [2,4,6]
inv_dimsub = 1 ./(Ls.^2)
n_mass = 5
n_alpha = 10
meas_array = zeros((length(Ls)[1], n_mass, n_alpha+1, 9))   # The for things which are stored are: {Δn, err_Δn, ϵ, mass, β}

intervalstep_tau = 1

Filename(L,i,mass) = string(file_folder(L),"/SublatticeSpin_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
File_phi(L,i, mass) = string(file_folder(L),"/Phi_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
File_S33_min_min(L,i, mass) = string(file_folder(L),"/S33_min_min_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
File_S3_plus(L,i, mass) = string(file_folder(L),"/S3_plus_interacting_eq",equation,"_m_",floor(Integer,mass*100),"_runlabel_",i)
ms = [0.1, 0.2, 0.3, 0.4, 0.5]
for L_ind = 1:length(Ls)[1]
    L = Ls[L_ind]
    for j = 1:length(ms)[1]
        for i = 1:n_alpha+1
            lat, HMC_par = Read_Settings(file_path_config(L,1), ["hmc", "lat"])
            if i<n_alpha+1
                par = Read_Settings(file_path_config(L,i), ["par"])
            else
                par = Read_Settings(file_path_config(L,10), ["par"])
            end

            if isfile(abspath(@__DIR__,string("../results/",File_S33_min_min(L,i,ms[j]),".txt")))
                meas_S33_min_min = ReadResult(File_S33_min_min(L,i,ms[j]))
                meas_S3_plus = ReadResult(File_S3_plus(L,i,ms[j]))
                meas_Δn = ReadResult(Filename(L,i,ms[j]))
                Δn_M = mean(meas_Δn[HMC_par.offset:intervalstep_tau:end])
                Δn2_M = mean(meas_Δn[HMC_par.offset:intervalstep_tau:end].^2)
                S3_plus = mean(meas_S3_plus[HMC_par.offset:intervalstep_tau:end])
                S3_plus2 = mean(meas_S3_plus[HMC_par.offset:intervalstep_tau:end].^2)
                S33_M = mean(meas_S33_min_min[HMC_par.offset:intervalstep_tau:end])
                S33_2_M = mean(meas_S33_min_min[HMC_par.offset:intervalstep_tau:end].^2)
                err_Δn_M = sqrt(abs(Δn2_M-Δn_M^2)/(floor((HMC_par.Nsamples-HMC_par.offset-1)/intervalstep_tau)))
                err_S3_plus = sqrt(abs(S3_plus2-S3_plus^2)/(floor((HMC_par.Nsamples-HMC_par.offset-1)/intervalstep_tau)))
                err_S33_M = sqrt(abs(S33_2_M-S33_M^2)/(floor((HMC_par.Nsamples-HMC_par.offset-1)/intervalstep_tau)))
                meas_array[L_ind, j, i, 1] = Δn_M
                meas_array[L_ind, j, i, 2] = err_Δn_M
                meas_array[L_ind, j, i, 3] = S33_M
                meas_array[L_ind, j, i, 4] = err_S33_M
                meas_array[L_ind, j, i, 5] = S3_plus
                meas_array[L_ind, j, i, 6] = err_S3_plus
            elseif i==n_alpha+1
                # create the M_function to be used for the computation of the observables.
                change_mass(ms[j], par)
                par.ϵ = 100000.0
                # @show par 
                # @show lat
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
                configurations = zeros((1,lat.D))
                meas = [S_33_min_min(-M_function(configurations[i,:]), par, lat) for i in 1:1] 
                meas_Δn = getindex.(meas, 1)
                meas_S33_min_min = getindex.(meas, 2)
                meas_S3_plus = getindex.(meas, 3)
                Δn_M = mean(meas_Δn)
                S33_M = mean(meas_S33_min_min)
                S3_plus = mean(meas_S3_plus)
                # @show S33_M
                # @show Δn_M
                # @show S3_M
                err_Δn_M = 0
                err_S33_M = 0
                err_S3_plus = 0
                meas_array[L_ind, j, i, 1] = Δn_M
                meas_array[L_ind, j, i, 2] = err_Δn_M
                meas_array[L_ind, j, i, 3] = S33_M
                meas_array[L_ind, j, i, 4] = err_S33_M
                meas_array[L_ind, j, i, 5] = S3_plus
                meas_array[L_ind, j, i, 6] = err_S3_plus
            end
            meas_array[L_ind, j, i, 7] = par.ϵ
            meas_array[L_ind, j, i, 8] = ms[j]   
            meas_array[L_ind, j, i, 9] = par.β 

        end
    end
end


αs = (300/137) ./meas_array[1, 1, :, 7]

volume_limit_folder = "Final_Volume_limit"



# we will now first perform a volume limit extrapolation, taking 1/dim_sub → 0
int_dimsub = [1,2,3]
Meas_extrapolate_array =  zeros((n_mass, n_alpha+1, 3)) 
for j = 1:length(ms)[1]
    for i = 1:length(αs)[1]
        fit = curve_fit(Polynomial, inv_dimsub[int_dimsub], meas_array[int_dimsub,j,i,1], 2)
        Meas_extrapolate_array[j,i,1] = fit.(0.0)
        fit = curve_fit(Polynomial, inv_dimsub[int_dimsub], meas_array[int_dimsub,j,i,3], 2)
        Meas_extrapolate_array[j,i,2] = fit.(0.0)
        fit = curve_fit(Polynomial, inv_dimsub[int_dimsub], meas_array[int_dimsub,j,i,5], 2)
        Meas_extrapolate_array[j,i,3] = fit.(0.0)
    end
end

m_x = 0.0:0.01:0.6
int_ms = [1, 2, 3, 4, 5]
int_αs = [1, 2, 3, 4, 5, 6, 7, 8, 11]

plot_extrapolate = true
extrapolate_alpha = [1, 3, 5, 7]
m_0_S33 = zeros(Float64,size(αs)[1])
if plot_extrapolate
    clf()
end

for i = 1:length(αs)[1]
    fit = curve_fit(Polynomial, ms[int_ms], Meas_extrapolate_array[int_ms,i,2], 2)
    y0b = fit.(m_x) 
    m_0_S33[i] = fit.(0.0)
    if plot_extrapolate 
        if i in extrapolate_alpha
            plot(ms,Meas_extrapolate_array[:,i,2],"o", label=string(L"\alpha_{eff}=",round(αs[i],digits=2))) 
            plot(m_x, y0b, "--", linewidth=1)
        end 
    end
end
if plot_extrapolate
    legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
    xlabel("mass")
    ylabel(L"$\langle S^{33}_{--} \rangle$")
    title(string(L"Mass extrapolation for $\langle S^{33}_{--} \rangle$"))
    savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S33_min_min_interacting_eq",equation,"_mass_extrapolation_multiple.png")))
end





m_0_S3_plus = zeros(Float64,size(αs)[1])
for i = 1:length(αs)[1]
    fit = curve_fit(Polynomial, ms[int_ms], Meas_extrapolate_array[int_ms,i,3], 2)
    y0b = fit.(m_x) 
    m_0_S3_plus[i] = fit.(0.0)
    if plot_extrapolate
        clf()
        plot(ms,Meas_extrapolate_array[:,i,3],"o") 
        plot(m_x, y0b, "--", linewidth=1)
        xlabel("mass")
        ylabel(L"$\langle S^{3}_{+} \rangle$")
        title(string(L"Mass extrapolation for $\alpha = $",round(αs[i], digits=3), L" $S^{3}_{+}(0) $= ",round(fit.(0.0), digits=3)))
        savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S3_plus_interacting_eq",equation,"_mass_extrapolation_runlabel_",i,".png")))
    end 
end

if plot_extrapolate
    clf()
end

m_0_S3_min = zeros(Float64,size(αs)[1])
for i = 1:length(αs)[1]
    fit = curve_fit(Polynomial, ms[int_ms], Meas_extrapolate_array[int_ms,i,1], 2)
    y0b = fit.(m_x) 
    m_0_S3_min[i] = fit.(0.0)
    if plot_extrapolate 
        if i in extrapolate_alpha
            plot(ms,Meas_extrapolate_array[:,i,1],"o", label=string(L"\alpha_{eff}=",round(αs[i],digits=2)))
            plot(m_x, y0b, "--", linewidth=1)
        end 
    end
end
if plot_extrapolate
    legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
    xlabel("mass")
    ylabel(L"$\langle S^{3}_{-} \rangle$")
    title(string(L"Mass extrapolation for $\langle S^{3}_{-} \rangle$"))
    savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S3_min_interacting_eq",equation,"_mass_extrapolation_multiple.png")))
end

clf()
for j in 1:length(ms)[1]
    plot(αs[int_αs], Meas_extrapolate_array[j,int_αs,1], "o", label=string("m = ",ms[j]))
end 
plot(αs[int_αs], m_0_S3_min[int_αs],"o",label=L"$m \rightarrow 0$")
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
# ylim([-0.1, maximum(ms)])
grid()
title(L"$\langle S^3_{-} \rangle$ after $N\rightarrow \infty$ extrapolation ")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{-} \rangle$")
savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S3_min_interacting_eq41_all_masses_with_extrpolate.png")))

clf()
for j in 1:length(ms)[1]
    plot(αs[int_αs], Meas_extrapolate_array[j,int_αs,2], "o", label=string("m = ",ms[j]))
end 
plot(αs[int_αs], m_0_S33[int_αs],"o",label=L"$m \rightarrow 0$")
legend(loc="center right", bbox_to_anchor=(1.1, 0.9))
# ylim([-0.1, maximum(ms)])
grid()
title(L"$\langle S^{33}_{--} \rangle$ after $N\rightarrow \infty$ extrapolation")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{33}_{--} \rangle$")
savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S33_min_min_interacting_eq41_all_masses_with_extrpolate.png")))


clf()
plot(αs[int_αs], m_0_S33[int_αs], "o")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{33}_{--} \rangle$")
title(string(L"$\langle S^{33}_{--}\rangle(m\rightarrow 0) $ for $\beta$ = ",round(meas_array[1,1, 1, 9],digits=3)))
savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S33_min_min_interacting_eq41_mass_extrapolation.png")))

clf()
plot(αs[int_αs], m_0_S3_plus[int_αs], "o")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{+} \rangle$")
title(string(L"$\langle S^{3}_{+} \rangle(m\rightarrow 0)$ for $\beta$ = ",round(meas_array[1,1, 1, 9],digits=3)))
savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S3_plus_interacting_eq41_mass_extrapolation.png")))

clf()
plot(αs[int_αs], m_0_S3_min[int_αs], "o")
xlabel(L"$\alpha_{eff}$")
ylabel(L"$\langle S^{3}_{-} \rangle$")
title(string(L"$\langle S^{3}_{-} \rangle(m\rightarrow 0)$ for $\beta$ =  ",round(meas_array[1,1, 1, 9],digits=3)))
savefig(abspath(@__DIR__,string("../results/",volume_limit_folder,"/S3_min_interacting_eq41_mass_extrapolation.png")))
