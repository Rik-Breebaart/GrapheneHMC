#= 
This script will contain the interacting Fermionic matrices.
=#
using LinearAlgebra
include("hexagonalLattice.jl")
include("tools.jl")
include("hamiltonianNoInteractions.jl")
#==================================== Matrix eq 35 ============================================#

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
    H = HamiltonianMatrix_no_int(par,lat) + Diagonal(V[diagind(V)])./2
    # performe permuted (δ_{t-1,t'}) on the array ensureing that the spatial part is correct
    M += δ*Array_Permute_Time_t1_t(H,lat) + im*ϕ.*kron(Diagonal(ones(2).*δ),kron(P,Diagonal(ones(lat.dim_sub))))
    return M
end 

function FermionicMatrix_int_35_saved_part(V, par::Parameters, lat::Lattice)
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
    H = HamiltonianMatrix_no_int(par,lat) + Diagonal(V[diagind(V)])./2
    # performe permuted (δ_{t-1,t'}) on the array ensureing that the spatial part is correct
    M += δ*Array_Permute_Time_t1_t(H,lat) 
    return M
end 

function FermionicMatrix_int_35_phi_part(ϕ, M_saved_part, par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    #create the permuation matrix for δ_{t-1,t'}
    P = time_permutation_Matrix_anti_pbc(lat)
    return M_saved_part + im*ϕ.*kron(Diagonal(ones(2).*δ),kron(P,Diagonal(ones(lat.dim_sub))))
end

#==================================== Matrix eq 41 ============================================#

function FermionicMatrix_int_41(ϕ, par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    #create the permuation matrix for δ_{t-1,t'}
    P = time_permutation_Matrix_anti_pbc(lat)
    # set the diagonal for (δ_{x,y}δ_{t,t'})
    M = Diagonal(ones(lat.D))
    # set the permutated of diagonal for (δ_{x,y}δ_{t-1,t'}) 
    # (split as shown below because the A and B sublattice are seperated in the matrix [first all A, then all B])
    M -= kron(Diagonal(ones(2)),kron(P,Diagonal(ones(lat.dim_sub))))*I.*exp.(-im*ϕ*δ)
    # determine the hamiltonian corresponding to the problem.
    H = HamiltonianMatrix_no_int(par,lat)
    # performe permuted (δ_{t-1,t'}) on the array ensureing that the spatial part is correct
    M += δ*Array_Permute_Time_t1_t(H,lat) 
    return M
end 

function FermionicMatrix_int_41_saved_part(par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    # set the diagonal for (δ_{x,y}δ_{t,t'})
    M = Diagonal(ones(lat.D))
    # determine the hamiltonian corresponding to the problem.
    H = HamiltonianMatrix_no_int(par,lat)
    # performe permuted (δ_{t-1,t'}) on the array ensureing that the spatial part is correct
    M += δ*Array_Permute_Time_t1_t(H,lat) 
    return M
end 


function FermionicMatrix_int_41_phi_part(ϕ, M_saved_part, par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    #create the permuation matrix for δ_{t-1,t'}
    P = time_permutation_Matrix_anti_pbc(lat)   
    # set the permutated of diagonal for (δ_{x,y}δ_{t-1,t'}) 
    # (split as shown below because the A and B sublattice are seperated in the matrix [first all A, then all B])
    return M_saved_part - kron(Diagonal(ones(2)),kron(P,Diagonal(ones(lat.dim_sub))))*I.*exp.(-im*ϕ*δ) 
end 