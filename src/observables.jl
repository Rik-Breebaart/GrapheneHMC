#=
    This file will contain the observables we will be looking at for graphene.
=#

include("hexagonalLattice.jl")
include("tools.jl")

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

function Δn(M, par::Parameters, lat::Lattice)
    P = time_permutation_Matrix_pbc(lat)
    # δ_{t-1, t'}δ_{x,y}m_x 
    # with m_x = 1 if x in A and -1 if x in B.
    S_P = kron(Diagonal([1,-1]),kron(P,Diagonal(ones(lat.dim_sub))))
    invM = S_P*inv(M)
    return (2/(lat.Nt*lat.dim_sub))*sum(diag(invM)[2:lat.D])
end