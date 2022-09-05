#= 
This script will contain the interactions matrices necessary for the graphene interacting model. 
The interactions are, coulomb, screened coulomb, partial screened and hubbard interactions.
=#

#To start we will introduce a simple Coulomb and hubbard interaction and look how the model reacts to these interactions.\

function coulomb_potential(par::Paramaters, lat::Lattice)
    