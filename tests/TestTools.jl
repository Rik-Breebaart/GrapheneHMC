

using LinearAlgebra, Test
include(abspath(@__DIR__,"../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))

function TestPermutation(par::Parameters, lat::Lattice)
    P = zeros((lat.Nt, lat.Nt))
    for t = 1:lat.Nt
        for tp = 1:lat.Nt
            t_min= t-1
            if tp==t_min #δ_{t',t-1}
                P[t,tp] = 1                
            elseif t_min==0 && tp==lat.Nt
                P[t,tp] = -1
            end 
        end 
    end 

    P_now = time_permutation_Matrix_anti_pbc(lat)
    @test P==P_now
end 


lat = Lattice(2,2,64)
par = Parameters(2.0, 0.0, 1.0, 0.5)
TestPermutation(par, lat)