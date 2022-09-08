#= This file contains the functions for the non-interactign system of graphene.
    We will first provide methods to compare the analytical results, and then 
    using the path-integral formalism on a discrete time lattice.

=#

using LinearAlgebra, IterativeSolvers
include("hexagonalLattice.jl")
include("tools.jl")

"""
Function to obtain the hamiltonian describing the tight binding model of graphene 
with a mass term with a sign difference between the two sublattices.

Input:
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)

Output:
    H  (Matrix of 2*Lm*Ln x 2*Lm*Ln Complex Floats) The Hamiltonian Matrix of Graphene without interactions
"""
function HamiltonianMatrix_no_int(par::Parameters, lat::Lattice)
    # positive mass for A sublattice and negative for B sublattice
    H = kron([[1, 0] [0, -1]],Diagonal(ones(lat.dim_sub)).*par.mass) -par.κ*neighbour_matrix(lat)
    # H = H_m + H_tb (for single spin)
    return H 
end 

function FermionicMatrix_no_int(par::Parameters,lat::Lattice)
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
    H = HamiltonianMatrix_no_int(par,lat)
    # performe permuted (δ_{t-1,t'}) on the array ensureing that the spatial part is correct
    M += δ*Array_Permute_Time_t1_t(H,lat)
    return M
end 
