#= 
This script will contain the interactions matrices necessary for the graphene interacting model. 
The interactions are, coulomb, screened coulomb, partial screened and hubbard interactions.
=#

include("hexagonalLattice.jl")
include("tools.jl")


#To start we will introduce a simple Coulomb and hubbard interaction and look how the model reacts to these interactions.\


"""

"""
function coulomb_potential(par::Parameters, lat::Lattice)
    # obtain the distance matrix r_ij =sqrt(ri^2+ri^2) (obeying periodic boundary conditions)
    r = distance_matrix(lat)
    # set the diagonal to 1 to ensure no devision by 0
    r[diagind(r)] .= 1
    Coulomb = (r).^(-1)
    Coulomb[diagind(Coulomb)] .= (par.R0*lat.a)^(-1)
    return Coulomb.*(par.e2/par.ϵ)
end 

"""

"""
function hubbard_potential(par::Parameters, lat::Lattice; self_interaction=9.3)
        return Diagonal(ones(lat.dim_sub*2).*self_interaction)
end 


"""

"""
function partialScreenedCoulomb(Rij, par::Parameters, lat::Lattice)
    U00 = 9.3
    U01 = 5.5
    U02 = 4.1
    U03 = 3.6
    m0 = [9.0380311, 2.0561977, 1.03347891, 1.0]
    m1 =1.0
    m2 = [144.354,27.8362, 0.0, 0.0]
    m3 = [62.41496, 15.29088, -0.134502, 0.0]
    γ = [0.632469, 0.862664, 0.990975, 1.0]
    eps = 10^(-8)
    if Rij>120*lat.a
        return (m3[4] .+ exp.(-m2[4].*Rij).*m0[4]./(m1.*Rij).^γ[4]).*par.e2
    elseif Rij>30*lat.a
        return (m3[3] .+ exp.(-m2[3].*Rij).*m0[3]./(m1.*Rij).^γ[3]).*par.e2
    elseif Rij>8*lat.a
        return (m3[2] .+ exp.(-m2[2].*Rij).*m0[2]./(m1.*Rij).^γ[2]).*par.e2
    elseif Rij>2*lat.a
        return (m3[1] .+ exp.(-m2[1].*Rij).*m0[1]./(m1.*Rij).^γ[1]).*par.e2
    elseif Rij>sqrt(3)*lat.a+eps
        return U03
    elseif Rij>1*lat.a+eps
        return U02
    elseif Rij> eps*lat.a+eps
        return U01
    else 
        return U00
    end
    return 0
end

"""

"""
function partialScreenedCoulomb_potential(par::Parameters, lat::Lattice)
    partial(r) = partialScreenedCoulomb(r, par, lat)
    r = distance_matrix(lat)
    V = map(partial,r)
    return V./par.ϵ
end 

"""

"""
function shortScreenedCoulomb(Rij, par::Parameters, lat::Lattice)
    U00 = 9.3
    U01 = 5.5
    U02 = 4.1
    U03 = 3.6
    e0 = par.e2/(lat.a*2*U03)
    eps = 10^(-8)
    
    if Rij>2*lat.a+eps
        return par.e2/(e0*Rij)
    elseif Rij>sqrt(3)*lat.a+eps
        return U03
    elseif Rij>1*lat.a+eps
        return U02
    elseif Rij> eps*lat.a+eps
        return U01
    else 
        return U00
    end
    return 0
end

"""

"""
function shortScreenedCoulomb_potential(par::Parameters, lat::Lattice)
    short(r) = shortScreenedCoulomb(r, par, lat)
    r = distance_matrix(lat)
    V = map(short,r)
    return V./par.ϵ
end


function basisSpaced_potential(V, par::Parameters, lat::Lattice)
    V = coulomb_potential(par, lat)
    V_k = fft(V)
    for x=1:2*lat.Lm
        for y = 1:3*lat.Ln
            V=V_k5
        end 
    end 
    return V_k
end 