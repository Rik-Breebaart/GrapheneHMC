#= 
This script will contain the interacting Fermionic matrices.
=#
using LinearAlgebra
include("hexagonalLattice.jl")
include("tools.jl")
include("hamiltonianNoInteractions.jl")
#==================================== Matrix eq 35 ============================================#

"""
Function to obtain the fermionic matrix M given by eq.35 from: 10.1103/PhysRevB.89.195429.

Input:
    ϕ   (Vector of D Floats)        Hubbard-Coulomb fields
    V   (Matrix of D x D Floats)    The Interaction potential between sites
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)

Output:
    M  (Matrix of D x D Complex Floats) The fermionic matrix
"""
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

"""
Function to obtain the part of the fermionic matrix M which does not depend on ϕ given by eq.35 from: 10.1103/PhysRevB.89.195429.
This matrix does thus not contain the ϕ related component

Input:
    V   (Matrix of D x D Floats)    The Interaction potential between sites
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)

Output:
    M_part  (Matrix of D x D Complex Floats) The fermionic matrix part independent of ϕ
"""
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


"""
Function to obtain the fermionic matrix M with numerical improved replacement
using the U(1) gauge invariance in the V_xx component of the Coulomb interaction.
(eg. eq.41 from: 10.1103/PhysRevB.89.195429)
 
Created as a matrix M + A[ϕ] where M is a supplied matrix which does not change during the simulations and A[ϕ] does change.
Input:
    ϕ   (Vector of D Floats)        Hubbard-Coulomb fields
    M_part (Matrix of D x D Complex Floats) The ϕ independent part of the fermionic matrix
    lat (Lattice struct)            Lattice struct containing the lattice paramaters
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)           

Output:
    M  (Matrix of D x D Complex Floats) The fermionic matrix
"""
function FermionicMatrix_int_35_phi_part(ϕ, M_saved_part, par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    #create the permuation matrix for δ_{t-1,t'}
    P = time_permutation_Matrix_anti_pbc(lat)
    return M_saved_part + im*ϕ.*kron(Diagonal(ones(2).*δ),kron(P,Diagonal(ones(lat.dim_sub))))
end

#==================================== Matrix eq 41 ============================================#

"""
Function to obtain the fermionic matrix M with numerical improved replacement
using the U(1) gauge invariance in the V_xx component of the Coulomb interaction.
(eg. eq.41 from: 10.1103/PhysRevB.89.195429)

Input:
    ϕ   (Vector of D Floats)        Hubbard-Coulomb fields
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)

Output:
    M  (Matrix of D x D Complex Floats) The fermionic matrix
"""
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

"""
Function to obtain constant component of the fermionic matrix M with numerical improved replacement
using the U(1) gauge invariance in the V_xx component of the Coulomb interaction. 
(eg. eq.41 from: 10.1103/PhysRevB.89.195429)

This matrix does thus not contain the ϕ related component

Input:
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)

Output:
    M_part  (Matrix of D x D Complex Floats) The fermionic matrix part independent of ϕ
"""
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


"""
Function to obtain the fermionic matrix M with numerical improved replacement
using the U(1) gauge invariance in the V_xx component of the Coulomb interaction.
(eg. eq.41 from: 10.1103/PhysRevB.89.195429)

Created as a matrix M + A[ϕ] where M is a supplied matrix which does not change during the simulations and A[ϕ] does change.
Input:
    ϕ   (Vector of D Floats)        Hubbard-Coulomb fields
    M_part (Matrix of D x D Complex Floats) The ϕ independent part of the fermionic matrix
    lat (Lattice struct)            Lattice struct containing the lattice paramaters
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)           

Output:
    M  (Matrix of D x D Complex Floats) The fermionic matrix
"""
function FermionicMatrix_int_41_phi_part(ϕ, M_saved_part, par::Parameters, lat::Lattice)
    # determine the temporal spacing
    δ=par.β/lat.Nt
    #create the permuation matrix for δ_{t-1,t'}
    P = time_permutation_Matrix_anti_pbc(lat)   
    # set the permutated of diagonal for (δ_{x,y}δ_{t-1,t'}) 
    # (split as shown below because the A and B sublattice are seperated in the matrix [first all A, then all B])
    return M_saved_part - kron(Diagonal(ones(2)),kron(P,Diagonal(ones(lat.dim_sub))))*I.*exp.(-im*ϕ*δ) 
end 