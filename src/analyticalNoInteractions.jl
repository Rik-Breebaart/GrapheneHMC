using LinearAlgebra
include("hexagonalLattice.jl")
include("tools.jl")

function greensFunctionGraphene_spatial(x,y,par,lat)
    v_a = sqrt(3)*lat.a*[1,0]
    v_b = sqrt(3)*lat.a*[1,sqrt(3)]/2
    k_a = (2*pi)/(3*lat.a)* [sqrt(3),-1]
    k_b = (4*pi)/(3*lat.a)* [0,1]
    δ= par.β/lat.Nt
    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b

    ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b
    Eu(x) = cos(x)+im*sin(x)
    Δ(k) = sum([Eu(dot(k,r_O[i,:])) for i=1:3])
    h(k) = [[par.mass,-par.κ*conj(Δ(k))] [-par.κ*Δ(k),-par.mass]]
    # invG(ω,k) = im*ω*I +h(k)
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)
    G(ω,k) = inv(invG(ω,k))

    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    ω(t) = pi*(2t+1)/par.β
    for τ=1:lat.Nt
        for m=1:lat.Lm, n=1:lat.Ln
            q = Eu(-dot(ks(m,n),(x-y)))
            for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
                correlator[τ,:,:] .+=G(ω(t),ks(m,n)).*q.*Eu(ω(t)*(τ-1/2)*(par.β/lat.Nt))
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
    δ= par.β/lat.Nt
    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b

    ks(m,n) = m/lat.Lm*k_a + n/lat.Ln * k_b
    Eu(x) = cos(x)+im*sin(x)
    Δ(k) = sum([Eu(dot(k,r_O[i,:])) for i=1:3])
    h(k) = [[par.mass,-par.κ*conj(Δ(k))] [-par.κ*Δ(k),-par.mass]]
    # invG(ω,k) = im*ω*I +h(k)
    invG(ω,k) = (2/δ)*im*Eu(ω*δ/2)*sin(ω*δ/2)*I +h(k)

    
    G(ω,k) = inv(invG(ω,k))

    correlator = zeros(ComplexF64,(lat.Nt,2,2))
    ω(t) = pi*(2t+1)/par.β
    for τ=1:lat.Nt
        for t = -floor(Int,lat.Nt/2):(floor(Int,lat.Nt/2)-1)
            correlator[τ,:,:] .+=G(ω(t),k).*Eu(ω(t)*(τ-1/2)*(par.β/lat.Nt))
        end 
    end 
    
    return correlator./(lat.Nt*par.β)
end 