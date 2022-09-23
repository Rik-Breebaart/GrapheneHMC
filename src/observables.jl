#=
    This file will contain the observables we will be looking at for graphene.
=#

include("hexagonalLattice.jl")
include("tools.jl")


"""
Greens function observable of the graphene system computed through the fermionic Matrix seperated in the four sublattice connections (AA, AB, BA, BB)

<G(τ)>_{x,y} = M^{-1}(x,y,τ,0)

Input:
    M   (Matrix lat.D x lat.D)  The fermionic matrix of the system 
    particle_x (Particle struct) Particle struct indicating the index coordinates of the x particle 
    particle_y (Particle struct) Particle struct indicating the index coordinates of the y particle 
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    correlator (lat.Ntx2x2)      The correlator greens function in space as a function of time and seperated on the 4 possible sublattice connections
"""
function greens_function_spatial(M, particle_x::Particle, particle_y::Particle, par::Parameters, lat::Lattice)
    M_inv = inv(M)
    correlator = zeros(ComplexF64, (lat.Nt,2,2))
    for Pab_x=[0,1]
        for Pab_y = [0,1]
            int_y_0 = index(particle_y.m, particle_y.n, 1, Pab_y, lat, start = particle_y.start)
            for τ=1:lat.Nt
                int_x_τ = index(particle_x.m, particle_x.n, τ, Pab_x, lat, start = particle_x.start)
                correlator[τ, Pab_x+1, Pab_y+1] = M_inv[int_x_τ, int_y_0]
            end
        end 
    end 
    return correlator
end 

"""
Greens function observable of the graphene system computed through the fermionic Matrix seperated in the four sublattice connections (AA, AB, BA, BB)
in k-space
THe fourier trasnforms are computed as 
F(x) = (1/N)∑f(k)exp{ikx}
F(t) = (1/β)∑f(ω)exp(iωt)

<G(τ)>_{k} = (1/N)∑ M^{-1}(x,y,τ,0)exp{-ik(x-y)}
with N = lat.dim_sub

Input:
    M   (Matrix lat.D x lat.D)      The fermionic matrix of the system 
    k   (2D vector of Floats)       The k-space coordinate to which we check the greens function
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
Output:
    correlator (lat.Ntx2x2)      The correlator greens function in k-space as a function of time and seperated on the 4 possible sublattice connections
"""
function greens_function_kspace(M, k, par::Parameters, lat::Lattice)
    M_inv = inv(M)
    Eu(x) = cos(x)+im*sin(x)
    correlator = zeros(ComplexF64, (lat.Nt,2,2))
    for Pab_x=[0,1]
        for Pab_y = [0,1]
            for m_x =1:lat.Lm, n_x = 1:lat.Ln
                for m_y =1:lat.Lm, n_y = 1:lat.Ln
                    int_y_0 = index(m_y, n_y, 1, Pab_y, lat, start = 1)
                    x = particle_position(m_x, n_x, Pab_x, lat, start=1)
                    y = particle_position(m_y, n_y, Pab_y, lat, start=1)
                    q = Eu(-dot(k,(x-y)))
                    for τ=1:lat.Nt
                        int_x_τ = index(m_x, n_x, τ, Pab_x, lat, start = 1)
                        correlator[τ, Pab_x+1, Pab_y+1] += M_inv[int_x_τ, int_y_0]*q
                    end 
                end                       
            end
        end 
    end 
    return correlator./(lat.dim_sub)
end 


"""
The sublattice spin difference of the graphene sublattices.
Δn  = n_A - n_B     With n_{A,x} = ϕ^{†}_x σ_z ϕ_x where σ_z is the third pauli matrix σ_z = [1,0; 0,-1]
and ϕ = 1/sqrt(2) (a_x↑, a_x↓)^T
Δn  = 2/(Nt*dim_sub)*∑_t Real(∑_{x∈A} M^{-1}_{x,t,x,t+1} - ∑_{x∈B} M^{-1}_{x,t,x,t+1} )
    = 2/(Nt*dim_sub)*Real(Trace(I_x δ_{t',t+1}δ_{x,y} M^{-1}))
    with I_x = 1 if x∈A and -1 if x∈B

Input:
    M   (Matrix lat.D x lat.D)      The fermionic matrix of the system 
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D) 
Output:
    Δn  (Float)                     The spin sublattice difference 
"""
function Δn(M, par::Parameters, lat::Lattice)
    P = time_permutation_Matrix_anti_pbc(lat)
    # δ_{t-1, t'}δ_{x,y}m_x 
    # with m_x = 1 if x in A and -1 if x in B.
    S_P = kron(Diagonal([1,-1]),kron(P,Diagonal(ones(lat.dim_sub))))
    invM = S_P*inv(M)
    return (2/(lat.Nt*lat.dim_sub))*real(sum(diag(invM)))
end


"""
The sublattice spin difference of the graphene sublattices as a functiuno of time.
Δn  = n_A - n_B     With n_{A,x} = ϕ^{†}_x σ_z ϕ_x where σ_z is the third pauli matrix σ_z = [1,0; 0,-1]
and ϕ = 1/sqrt(2) (a_x↑, a_x↓)^T
Δn  = 2/(Nt*dim_sub)*Real(∑_{x∈A} M^{-1}_{x,t,x,t+1} - ∑_{x∈B} M^{-1}_{x,t,x,t+1} )
    = 2/(Nt*dim_sub)*Real(Trace(I_x δ_{t',t+1}δ_{x,y} M^{-1}))
    with I_x = 1 if x∈A and -1 if x∈B

Input:
    M   (Matrix lat.D x lat.D)      The fermionic matrix of the system 
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D) 
Output:
    Δn(τ)  (lat.Nt Floats)           The spin sublattice difference as a function of time
"""
function Δn_time(M, par::Parameters, lat::Lattice)
    M_inv = inv(M)
    correlator = zeros(ComplexF64, (lat.Nt,2,2))
    for Pab_x=[0,1]
        for Pab_y = [0,1]
            int_y_0 = index(1, 1, 1, Pab_y, lat, start = 1)
            for τ=1:lat.Nt
                int_x_τ = index(1, 1, τ, Pab_x, lat, start = 1)
                correlator[τ, Pab_x+1, Pab_y+1] = M_inv[int_x_τ, int_y_0]
            end
        end 
    end 
    return 2*(correlator[:,1,1]-correlator[:,2,2])
end


"""
The average of the Hubbard Stratonovich Field on the Pab sublattice. Aimed to look at thermalization of the hmc sweeps
Input:
    ϕ   (Complex lat.D)             The Hubbard Stratonovich Field
    Pab (Int (eiother 0 or 1))      The boolean indicating on which sublattice we look (Pab=0,1 is A,B sublatice respectivly.) 
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D) 
Output:
    mean(ϕ) (Complex Float)         The average Hubbard Stratonovich Field on the Pab sublatice
"""
function HubbardStratonovichField(ϕ, Pab, par::Parameters, lat::Lattice)
    ind_part = lat.dim_sub*lat.Nt
    if Pab==0
        return mean(ϕ[1:ind_part])
    elseif Pab==1
        return mean(ϕ[ind_part+1:lat.D])
    else 
        error("An incorrect sublattice type index is given, Pab should be either 0 or 1.")
    end 
    return 0
end 