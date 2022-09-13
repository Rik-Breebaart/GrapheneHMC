
using LinearAlgebra, Statistics
include("hamiltonianInteractions.jl")
include("hamiltonianNoInteractions.jl")
include("interactions.jl")
include("tools.jl")
"""
Action for the interacting tight binding model of graphene with psuedo bosonic field χ to compute the fermionic determinants. 
    S_χ = χ^{†}(MM^{†})^{-1}χ
"""
function Action_M_cg(χ, M, par::Parameters, lat::Lattice)
    η = cg(M*adjoint(M), χ) 
    return adjoint(χ)*η
end 

function Action_M(χ, M, par::Parameters, lat::Lattice)
    ρ = M \ χ
    return adjoint(ρ)*ρ
end 

function Action_V(ϕ, V, par::Parameters, lat::Lattice)
    invV = Array_Equal_Time(inv(V), lat)
    return (par.β/(lat.Nt*2))*transpose(ϕ)*invV*ϕ
end 

function Action_V_cg(ϕ, V, par::Parameters, lat::Lattice)
    V_equal_time =  Array_Equal_Time(V, lat)
    η = cg(V_equal_time, ϕ)
    return (par.β/(lat.Nt*2))*transpose(ϕ)*η
end 

function ∇S_M_eq35_cg(ϕ, χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = cg(M*adjoint(M), χ) 
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag(conj(η).*P*adjoint(M)*η)
end 

function ∇S_M_eq35(ϕ, χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = inv(M*adjoint(M))*χ
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag(conj(η).*P*adjoint(M)*η)
end 

function ∇S_V(ϕ, V, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    V_time = Array_Equal_Time(V, lat)
    return  δ.*transpose(inv(V_time))*ϕ 
end 

function ∇S_V_cg(ϕ, V, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    V_equal_time =  Array_Equal_Time(V, lat)
    η = cg(transpose(V_equal_time), ϕ)
    return δ*η
end 

function ∇S_M_eq41(ϕ ,χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = inv(M*adjoint(M))*χ
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag((diagm(ones(lat.D)).*exp.(-im*δ*ϕ)*conj(η)).*(P*adjoint(M)*η))
end 

function ∇S_M_eq41_cg(ϕ ,χ, M, par::Parameters, lat::Lattice)
    δ = par.β/lat.Nt
    η = cg(M*adjoint(M), χ) 
    P = kron(Diagonal(ones(2)),kron(time_permutation_Matrix_anti_pbc(lat),Diagonal(ones(lat.dim_sub))))  
    return (2*δ).*imag((diagm(ones(lat.D)).*exp.(-im*δ*ϕ)*conj(η)).*(P*adjoint(M)*η))
end 