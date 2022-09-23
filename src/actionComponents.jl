
using LinearAlgebra, Statistics
include("hamiltonianInteractions.jl")
include("hamiltonianNoInteractions.jl")
include("interactions.jl")
include("tools.jl")

"""
Action for the interacting tight binding model of graphene with psuedo bosonic field χ to compute the fermionic determinants. 
    S_χ = χ^{†}(MM^{†})^{-1}χ
In this function the inverse is computed using conjugate gradient methods.

Input: 
    χ (Complex Lat.D vector)        The psuedo bosonix complex χ field 
    M (complex lat.D x lat.D)       The Fermionic matrix of the system
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    S_M (Floats)                    The action xomponent corresponding to the fermionic matrix part
"""
function Action_M_cg(χ, M, par::Parameters, lat::Lattice)
    η = cg(M*adjoint(M), χ) 
    return adjoint(χ)*η
end 


"""
Action for the interacting tight binding model of graphene with psuedo bosonic field χ to compute the fermionic determinants. 
    S_χ = χ^{†}(MM^{†})^{-1}χ
In this function the inverse is computed using the LinearAlgebra inverse function.
    
Input: 
    χ (Complex Lat.D vector)        The psuedo bosonix complex χ field 
    M (complex lat.D x lat.D)       The Fermionic matrix of the system
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    S_M (Floats)                    The action xomponent corresponding to the fermionic matrix part
"""
function Action_M(χ, M, par::Parameters, lat::Lattice)
    ρ = M \ χ
    return adjoint(ρ)*ρ
end 

"""
Action for the potential component of the action of the tight binding model of graphene with interactions.
    S_V = (δ/2)ϕ^T V^{-1} ϕ
In this function the inverse is computed using the LinearAlgebra inverse function.
    
Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    V (Floats lat.dim_sub*2 x lat.dim_sub*2)    The potential matrix of the interactions between the different sites.
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    S_V (Floats)                    The action xomponent corresponding to the fermionic matrix part
"""
function Action_V(ϕ, V, par::Parameters, lat::Lattice)
    invV = Array_Equal_Time(inv(V), lat)
    return (par.β/(lat.Nt*2))*transpose(ϕ)*invV*ϕ
end 

"""
Action for the potential component of the action of the tight binding model of graphene with interactions.
    S_V = (δ/2)ϕ^T V^{-1} ϕ
In this function the inverse is computed using conjugate gradient methods.

Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    V (Floats lat.dim_sub*2 x lat.dim_sub*2)    The potential matrix of the interactions between the different sites.
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    S_V (Floats)                    The action xomponent corresponding to the fermionic matrix part
"""
function Action_V_cg(ϕ, V, par::Parameters, lat::Lattice)
    V_equal_time =  Array_Equal_Time(V, lat)
    η = cg(V_equal_time, ϕ)
    return (par.β/(lat.Nt*2))*transpose(ϕ)*η
end 

#====================={Gradient components for hamiltonian integration path}=====================#
"""
Gradient component of the fermionic matrix part of the action for the fermionic matrix computed 
through equation 35 of ref{Arxiv:10.1103/PhysRevB.89.195429}.
∇S_M = 2δIm[conj(η)_{x,t}*ξ_{x,t-1}]    with η=(MM^{†})^{-1}χ and ξ=m^{-1}χ=M^{†}η.

In this function the inverse is computed using conjugate gradient methods.

Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    χ (Complex Lat.D vector)        The psuedo bosonix complex χ field 
    M (complex lat.D x lat.D)       The Fermionic matrix of the system
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    ∇S_M (Complex lat.D)            The gradient of the action component corresponding to the fermionic matrix eq.35
"""
function ∇S_M_eq35_cg(ϕ, χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = cg(M*adjoint(M), χ) 
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag(conj(η).*P*adjoint(M)*η)
end 

"""
Gradient component of the fermionic matrix part of the action for the fermionic matrix computed 
through equation 35 of ref{Arxiv:10.1103/PhysRevB.89.195429}.
∇S_M = 2δIm[conj(η)_{x,t}*ξ_{x,t-1}]    with η=(MM^{†})^{-1}χ and ξ=m^{-1}χ=M^{†}η.

In this function the inverse is computed using the LinearAlgebra inverse function.
    
Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    χ (Complex Lat.D vector)        The psuedo bosonix complex χ field 
    M (complex lat.D x lat.D)       The Fermionic matrix of the system
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    ∇S_M (Complex lat.D)            The gradient of the action component corresponding to the fermionic matrix eq.41
"""
function ∇S_M_eq35(ϕ, χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = inv(M*adjoint(M))*χ
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag(conj(η).*P*adjoint(M)*η)
end 

"""
Gradient component of the fermionic matrix part of the action for the fermionic matrix computed 
through equation 41 of ref{Arxiv:10.1103/PhysRevB.89.195429}.
∇S_M = 2δIm[conj(η)_{x,t}*exp{-im δ ϕ_{x,t}}*ξ_{x,t-1}]    with η=(MM^{†})^{-1}χ and ξ=m^{-1}χ=M^{†}η.

In this function the inverse is computed using conjugate gradient methods.
  
Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    χ (Complex Lat.D vector)        The psuedo bosonix complex χ field 
    M (complex lat.D x lat.D)       The Fermionic matrix of the system
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    ∇S_M (Complex lat.D)            The gradient of the action component corresponding to the fermionic matrix eq.41
"""
function ∇S_M_eq41(ϕ ,χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = inv(M*adjoint(M))*χ
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag((diagm(ones(lat.D)).*exp.(-im*δ*ϕ)*conj(η)).*(P*adjoint(M)*η))
end 

"""
Gradient component of the fermionic matrix part of the action for the fermionic matrix computed 
through equation 41 of ref{Arxiv:10.1103/PhysRevB.89.195429}.
∇S_M = 2δIm[conj(η)_{x,t}*exp{-im δ ϕ_{x,t}}*ξ_{x,t-1}]    with η=(MM^{†})^{-1}χ and ξ=m^{-1}χ=M^{†}η.

In this function the inverse is computed using the LinearAlgebra inverse function.
    
Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    χ (Complex Lat.D vector)        The psuedo bosonix complex χ field 
    M (complex lat.D x lat.D)       The Fermionic matrix of the system
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    ∇S_M (Complex lat.D)            The gradient of the action component corresponding to the fermionic matrix eq.35
"""
function ∇S_M_eq41_cg(ϕ ,χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = cg(M*adjoint(M), χ) 
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag((diagm(ones(lat.D)).*exp.(-im*δ*ϕ)*conj(η)).*(P*adjoint(M)*η))
end 

"""
Gradient of the action component related to the potential part of the tight binding model of graphene with interactions.
∇S_V = δ ϕ^{T} V_{-1}

In this function the inverse is computed using the LinearAlgebra inverse function.
    
Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    V (Floats lat.dim_sub*2 x lat.dim_sub*2)    The potential matrix of the interactions between the different sites.
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    ∇S_V (Floats)                    The gradient of the action component corresponding to the potential part
"""
function ∇S_V(ϕ, V, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    V_time = Array_Equal_Time(V, lat)
    return  δ.*transpose(inv(V_time))*ϕ 
end 


"""
Gradient of the action component related to the potential part of the tight binding model of graphene with interactions.
∇S_V = δ ϕ^{T} V_{-1} = δ V^{-1 ,T }ϕ

In this function the inverse is computed using conjugate gradient methods.
    
Input: 
    ϕ (Complex Lat.D vector)        The hubbard stratonivich field ϕ
    V (Floats lat.dim_sub*2 x lat.dim_sub*2)    The potential matrix of the interactions between the different sites.
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    ∇S_V (Floats)                    The gradient of the action component corresponding to the potential part
"""
function ∇S_V_cg(ϕ, V, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    V_equal_time =  Array_Equal_Time(V, lat)
    η = cg(transpose(V_equal_time), ϕ)
    return δ*η
end 
