
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
lat = Lattice(2, 2, 8)
ms = [0.5, 0.4, 0.3, 0.2, 0.1]
αs = LinRange(0.1, 5.0, 11)
ϵs = (300/137)./αs
rng = MersenneTwister(123)

path_length = 10.0
step_size = 0.6
m = 5 #sexton weingarten split Fermionic substeps
Nsamples= 10000
burn_in = 100
offset = floor(Integer, Nsamples*0.2)
β = 2.0
par = Parameters(β, ms[1], ϵs[1], 0.5)
HMC_par = HMC_Parameters(Nsamples, path_length, step_size, offset, m, burn_in)

file_path = abspath(@__DIR__, "../test_folder/test_config.csv")
Store_Settings(file_path, HMC_par)
Store_Settings(file_path, lat, method="a")
Store_Settings(file_path, par, method="a")
