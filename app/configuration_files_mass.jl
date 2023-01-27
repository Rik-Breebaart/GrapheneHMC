
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
lat = Lattice(2, 2, 16)
# αs = LinRange(0.1, 5.0, 2)
equation = 41
# αs = LinRange(0.45, (300/137), 6)
# ϵs = (300/137)./αs
ϵs =  LinRange(1.3, 0.1, 10)
# ϵs = (300/137)./αs[1:5]

filename = "run"
path_length = 10.0
step_size = 0.5
m = 5 #sexton weingarten split Fermionic substeps
Nsamples= 5000
burn_in = 100
offset = floor(Integer, Nsamples*0.3)
T = 1/2
β = 1/T
HMC_par = HMC_Parameters(Nsamples, path_length, step_size, offset, m, burn_in, equation)


folder = string("SublatticeSpinDifference_", lat.Lm, "_", lat.Ln,"_Nt_", lat.Nt, "_eq",equation,"_final") #or use storrage_folder function
conf_folder = "configurations"
subfolder = "Intermediate_results"
extrapolate_folder = "extrapolate"
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
if isdir(sub_folder(extrapolate_folder))==false
    mkdir(sub_folder(extrapolate_folder))
end

for i in 1:length(ϵs)[1]
    par = Parameters(β, 0.0 , ϵs[i], 0.5)
    file_path = abspath(@__DIR__,string("../results/",folder,"/",conf_folder,"/",filename,"_$i.csv"))
    Store_Settings(file_path, HMC_par)
    Store_Settings(file_path, lat, method="a")
    Store_Settings(file_path, par, method="a")
end
