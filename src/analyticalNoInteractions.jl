using LinearAlgebra
include("hexagonalLattice.jl")
include("tools.jl")

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
    # invG(ω,k) = im*ω*I +h(k)
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)
    # invG(ω,k) = [[(2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2), 0] [0, (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)]] +h(k)


    G(ω,k) = inv(invG(ω,k))

    x(Pab) = particle_position(particle_x.m, particle_x.n, Pab, lat, start=particle_x.start)
    y(Pab) = particle_position(particle_y.m, particle_y.n, Pab, lat, start=particle_y.start)
    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    ω(t) = pi*(2t+1)/par.β
    q = zeros(ComplexF64, (2,2))
    for τ=1:lat.Nt
        for m=1:lat.Lm, n=1:lat.Ln
            for Pab_x = [0,1], Pab_y =[0,1]
                q[Pab_x+1,Pab_y+1] = Eu(dot(ks(m,n),(y(Pab_x)-x(Pab_y))))
            end 
            for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
                correlator[τ,:,:] .+=G(ω(t),ks(m,n))*q.*Eu(ω(t)*(τ-1)*δ)
            end 
        end 
    end 
    
    return correlator./(par.β*lat.Nt)
end 



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
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)
    # invG(ω,k) = [[(2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2), 0] [0, (2/δ)*im*Eu(-ω*δ/2)*sin(ω*δ/2)]] +h(k)
    

    G(ω,k) = inv(invG(ω,k))

    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    ω(t) = pi*(2t+1)/par.β
    for τ=1:lat.Nt
        for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
            correlator[τ,:,:] .+=G(ω(t),k).*Eu(ω(t)*(τ-1)*δ)
        end 
    end 
    
    return correlator./(lat.Nt*par.β)
end 
