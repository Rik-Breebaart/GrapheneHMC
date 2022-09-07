#= 
This script will contain the interactions matrices necessary for the graphene interacting model. 
The interactions are, coulomb, screened coulomb, partial screened and hubbard interactions.
=#

include("hexagonalLattice.jl")
include("tools.jl")


#To start we will introduce a simple Coulomb and hubbard interaction and look how the model reacts to these interactions.\



function coulomb_potential(par::Parameters, lat::Lattice)
    # obtain the distance matrix r_ij =sqrt(ri^2+ri^2) (obeying periodic boundary conditions)
    r = distance_matrix(lat)
    # set the diagonal to 1 to ensure no devision by 0
    r[diagind(r)] .= 1
    Coulomb = (r).^(-1)
    Coulomb[diagind(Coulomb)] .= (par.R0*lat.a)^(-1)
    return Coulomb.*(par.e2/par.ϵ)
end 
    