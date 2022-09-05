#= 
This script will contain the interacting fermionic matrices.
=#
using LinearAlgebra
include("hexagonalLattice.jl")
include("tools.jl")
include("hamiltonianNoInteractions.jl")



function FermionicMatrix_int_35(ϕ, V, par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    #create the permuation matrix for δ_{t-1,t'}
    P = time_permutation_Matrix_anti_pbc(lat)
    # set the diagonal for (δ_{x,y}δ_{t,t'})
    M = Diagonal(ones(lat.D))
    # set the permutated of diagonal for (δ_{x,y}δ_{t-1,t'}) 
    # (split as shown below because the A and B sublattice are seperated in the matrix [first all A, then all B])
    M -= kron(Diagonal(ones(2)),kron(P,Diagonal(ones(lat.dim_sub))))
    # determine the hamiltonian corresponding to the problem.
    H = HamiltonianMatrix_no_int(par,lat) + diagonal(V)
    # performe permuted (δ_{t-1,t'}) on the array ensureing that the spatial part is correct
    M += δ*Array_Permute_Time_t1_t(H,lat)
    return M
end 
