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
lat = Lattice(6, 6, 1)
Nts = [8, 12, 16]

αs = 4.00
ϵs = (300/137)/αs

ms = 0.5
equation = 41
folder = string("Continuum_limit_", lat.Lm, "_", lat.Ln,"_m_",floor(Integer,ms*10),"_alpha_",floor(Integer,αs*100),"_eq",equation)#or use storrage_folder functionx

filename = "run"
path_length = 10.0
step_size = 0.5
m = 5 #sexton weingarten split Fermionic substeps
Nsamples= 5000
burn_in = 100
offset = floor(Integer, Nsamples*0.4)
β = 2.0
HMC_par = HMC_Parameters(Nsamples, path_length, step_size, offset, m, burn_in, equation)
par = Parameters(β, ms, ϵs, 0.5)

conf_folder = "configurations"
subfolder = "Intermediate_results"
Thermalization_folder = "thermalization"
folder_full_dir = abspath(@__DIR__,string("../results/",folder))
sub_folder(new_folder) = abspath(@__DIR__,string("../results/",folder,"/",new_folder))

if isdir(folder_full_dir)==false
    mkdir(folder_full_dir)
end 
if isdir(sub_folder(subfolder))==false
    mkdir(sub_folder(subfolder))
end
if isdir(sub_folder(conf_folder))==false
    mkdir(sub_folder(conf_folder))
end
if isdir(sub_folder(Thermalization_folder))==false
    mkdir(sub_folder(Thermalization_folder))
end

for i in 1:length(Nts)[1]
    change_lat(lat, Nt = Nts[i])
    file_path = abspath(@__DIR__,string("../results/",folder,"/",conf_folder,"/",filename,"_$i.csv"))
    Store_Settings(file_path, HMC_par)
    Store_Settings(file_path, lat, method="a")
    Store_Settings(file_path, par, method="a")
end
