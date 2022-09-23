using LinearAlgebra
include("hexagonalLattice.jl")
include("tools.jl")


"""
Analytical solution for the greens function of graphene in the spatial domain. 
The greens function is split into the sublattice connections (AA,AB,BA,BB)
The basis of the vectors that is used is the basis as described in (Arxiv:1403.3620)
The K space basis is the reciprocal of the position space basis scaled within the brilouine zone.

The hamiltonian of the system is given in k-space and then converted to real space through a discrete fourier tranform 
F(x) = (1/sqrt(N))∑f(k)exp{ikx}
F(t) = (1/sqrt(β))∑f(ω)exp(iωt)

For the temporal derivative in matsubara frequencies a discrete forward derivative is used converted to matsubara 
frequency space using fourier transform.
G = inv(f(ω) + h(k))


Input: 
    particle_x  (Particle)  particle for which the position is used
    particle_y  (Particle)  particle for which the position is used
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
Output: 
    correlator (4xlat.Nt)   The greens function in position space.

"""
function greensFunctionGraphene_spatial(particle_x::Particle, particle_y::Particle, par::Parameters, lat::Lattice)
    v_a = sqrt(3)*lat.a*[1,0]
    v_b = sqrt(3)*lat.a*[1,sqrt(3)]/2

    k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
    k_b = (4*pi)/(3*lat.a)* [0,1]
    ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b

    δ= par.β/lat.Nt

    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b

    Eu(x) = cos(x)+im*sin(x)
    Δ(k) = sum([Eu(dot(k,r_O[i,:])) for i=1:3])
    h(k) = [[par.mass,-par.κ*conj(Δ(k))] [-par.κ*Δ(k),-par.mass]]
    # invG(ω,k) = im*ω*I +h(k)      #continues limit
    # invG(ω,k) = (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)*I +h(k)      #discrete backward
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)     #discrete forward
    # invG(ω,k) = [[(2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2), 0] [0, (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)]] +h(k)        #discrete mixed


    G(ω,k) = inv(invG(ω,k))

    x(Pab) = particle_position(particle_x.m, particle_x.n, Pab, lat, start=particle_x.start)
    y(Pab) = particle_position(particle_y.m, particle_y.n, Pab, lat, start=particle_y.start)
    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    ω(t) = pi*(2t+1)/par.β
    q = zeros(ComplexF64, (2,2))
    for τ=1:lat.Nt
        for m=1:lat.Lm, n=1:lat.Ln
            for Pab_x = [0,1], Pab_y =[0,1]
                q[Pab_x+1,Pab_y+1] = Eu(dot(ks(m,n),(x(Pab_x)-y(Pab_y))))
            end 
            for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
                correlator[τ,:,:] .+=G(ω(t),ks(m,n)).*q.*Eu(ω(t)*(τ)*δ)
            end 
        end 
    end 
    
    return correlator./(par.β*lat.dim_sub)
end 


"""
Analytical solution for the greens function of graphene in the spatial domain. 
The greens function is split into the sublattice connections (AA,AB,BA,BB)
The basis of the vectors that is used is the basis as described in (Arxiv:1403.3620)
The K space basis is the reciprocal of the position space basis scaled within the brilouine zone.

The hamiltonian of the system is given in k-space and then converted to real space through a discrete fourier tranform 
F(x) = (1/sqrt(N))∑f(k)exp{ikx}
F(t) = (1/sqrt(β))∑f(ω)exp(iωt)

For the temporal derivative in matsubara frequencies a discrete forward derivative is used converted to matsubara 
frequency space using fourier transform.
G = inv(f(ω) + h(k))


Input: 
    K (Float vector of 2)           The vector of momentum coordinates to which the greens function is computed
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
Output: 
    correlator (4xlat.Nt)   The greens function in momentum space (and time).
    
"""
function greensFunctionGraphene_kspace(k,par,lat)
    v_a = sqrt(3)*lat.a*[1,0]
    v_b = sqrt(3)*lat.a*[1,sqrt(3)]/2

    k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
    k_b = (4*pi)/(3*lat.a)* [0,1]
    ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b

    δ= par.β/lat.Nt

    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b

    Eu(x) = cos(x)+im*sin(x)
    Δ(k) = sum([Eu(dot(k,r_O[i,:])) for i=1:3])
    h(k) = [[par.mass,-par.κ*conj(Δ(k))] [-par.κ*Δ(k),-par.mass]]
    # invG(ω,k) = im*ω*I +h(k)
    # invG(ω,k) = (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)*I +h(k)
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)
    # invG(ω,k) = [[(2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2), 0] [0, (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)]] +h(k)
    

    G(ω,k) = inv(invG(ω,k))

    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    ω(t) = pi*(2t+1)/par.β
    for τ=1:lat.Nt
        for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
            correlator[τ,:,:] .+=G(ω(t),k).*Eu(ω(t)*(τ)*δ)
        end 
    end 
    
    return correlator./(par.β)
end 


"""
Analytical solution for the sublatice spin difference of graphene.
This is computed through the correlator greens functions in space.
The greens function is split into the sublattice connections (AA,AB,BA,BB)
The basis of the vectors that is used is the basis as described in (Arxiv:1403.3620)
The K space basis is the reciprocal of the position space basis scaled within the brilouine zone.

The hamiltonian of the system is given in k-space and then converted to real space through a discrete fourier tranform 
F(x) = (1/N)∑f(k)exp{ikx}
F(t) = (1/β)∑f(ω)exp(iωt)

For the temporal derivative in matsubara frequencies a discrete forward derivative is used converted to matsubara 
frequency space using fourier transform.
G = inv(f(ω) + h(k))

then 
Δn(τ) = 2*(G_AA(τ) - G_BB(τ))
where it is multiplied by to since the fermionic hmc observable contains both spin up and down degrees of freedom 
with the same greens function.


Input: 
    K (Float vector of 2)           The vector of momentum coordinates to which the greens function is computed
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
Output: 
    Δn(τ) (lat.Nt)                  The sublatice spin difference as a function of τ
    
"""
function Δn_no_int_time(par::Parameters, lat::Lattice)
    v_a = sqrt(3)*lat.a*[1,0]
    v_b = sqrt(3)*lat.a*[1,sqrt(3)]/2

    k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
    k_b = (4*pi)/(3*lat.a)* [0,1]
    ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b

    δ= par.β/lat.Nt

    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b

    Eu(x) = cos(x)+im*sin(x)
    Δ(k) = sum([Eu(dot(k,r_O[i,:])) for i=1:3])
    h(k) = [[par.mass,-par.κ*conj(Δ(k))] [-par.κ*Δ(k),-par.mass]]
    # invG(ω,k) = im*ω*I +h(k)
    invG(ω,k) = (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)*I +h(k)
    # invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)
    # invG(ω,k) = [[(2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2), 0] [0, (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)]] +h(k)

    G(ω,k) = inv(invG(ω,k))

    mx =1
    nx=1
    xy = zeros((2,2,2))
    for Pab_x = [0,1], Pab_y = [0,1]
        x = particle_position(mx,nx,Pab_x,lat,start=1)
        y = particle_position(mx,nx,Pab_y,lat,start=1)
        xy[Pab_x+1,Pab_y+1,:] = x-y
    end
    ω(t) = pi*(2t+1)/par.β
    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    q = zeros(ComplexF64, (2,2))
    for τ=1:lat.Nt
        for m=1:lat.Lm, n=1:lat.Ln
            for Pab_x = [0,1], Pab_y =[0,1]
                q[Pab_x+1,Pab_y+1] = Eu(dot(ks(m,n),(xy[Pab_x+1,Pab_y+1,:])))
            end 
            for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
                correlator[τ,:,:] .+=G(ω(t),ks(m,n)).*q.*Eu(ω(t)*(τ)*δ)
            end 
        end 
    end 
    Δn = 2*(correlator[:,1,1]-correlator[:,2,2])
    return Δn./(par.β*lat.dim_sub)
end 

"""
Analytical solution for the sublatice spin difference of graphene.
This is computed through the correlator greens functions in space.
The greens function is split into the sublattice connections (AA,AB,BA,BB)
The basis of the vectors that is used is the basis as described in (Arxiv:1403.3620)
The K space basis is the reciprocal of the position space basis scaled within the brilouine zone.

The hamiltonian of the system is given in k-space and then converted to real space through a discrete fourier tranform 
F(x) = (1/N)∑f(k)exp{ikx}
F(t) = (1/β)∑f(ω)exp(iωt)

For the temporal derivative in matsubara frequencies a discrete forward derivative is used converted to matsubara 
frequency space using fourier transform.
G = inv(f(ω) + h(k))

then 
Δn = 2*(G_AA - G_BB)
where it is multiplied by to since the fermionic hmc observable contains both spin up and down degrees of freedom 
with the same greens function.


Input: 
    K (Float vector of 2)           The vector of momentum coordinates to which the greens function is computed
    lat (Lattice struct)            Lattice struct containing the lattice paramaters (Lm, LN, Nt, a, dim_sub, D)
    par (Paramaters struct)         Paramaters struct containing the run paramaters (α, β etc.)
Output: 
    Δn                              The sublatice spin difference
"""        
function Δn_no_int(par::Parameters, lat::Lattice)
    v_a = sqrt(3)*lat.a*[1,0]
    v_b = sqrt(3)*lat.a*[1,sqrt(3)]/2

    k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
    k_b = (4*pi)/(3*lat.a)* [0,1]
    ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b

    δ= par.β/lat.Nt

    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b

    Eu(x) = cos(x)+im*sin(x)
    Δ(k) = sum([Eu(dot(k,r_O[i,:])) for i=1:3])
    h(k) = [[par.mass,-par.κ*conj(Δ(k))] [-par.κ*Δ(k),-par.mass]]
    # invG(ω,k) = im*ω*I +h(k)      #continues limit
    # invG(ω,k) = (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)*I +h(k)      #discrete backward
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)     #discrete forward
    # invG(ω,k) = [[(2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2), 0] [0, (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)]] +h(k)        #discrete mixed

    G(ω,k) = inv(invG(ω,k))

    mx =1
    nx=1
    xy = zeros((2,2,2))
    for Pab_x = [0,1], Pab_y = [0,1]
        x = particle_position(mx,nx,Pab_x,lat,start=1)
        y = particle_position(mx,nx,Pab_y,lat,start=1)
        xy[Pab_x+1,Pab_y+1,:] = x-y
    end
    ω(t) = pi*(2t+1)/par.β
    correlator = zeros(ComplexF64,(2,2))
    q = zeros(ComplexF64, (2,2))
    for m=1:lat.Lm, n=1:lat.Ln
        for Pab_x = [0,1], Pab_y =[0,1]
            q[Pab_x+1,Pab_y+1] = Eu(dot(ks(m,n),(xy[Pab_x+1,Pab_y+1,:])))
        end 
        for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
            correlator[:,:] .+=G(ω(t),ks(m,n)).*q
        end 
    end 
    Δn = 2*(correlator[1,1]-correlator[2,2])
    return Δn./(par.β*lat.dim_sub)
end 