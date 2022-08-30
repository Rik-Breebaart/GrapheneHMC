
using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot


Lm = 8
Ln = 8
Nt = 20
mass = 0.0
a=1
β = 2.0
int(x) = floor(Int, x)

v_a = sqrt(3)*a*[1,0]
v_b = sqrt(3)*a*[-1,sqrt(3)]/2
k_a = (2*pi)/(3*a)* [sqrt(3),1]
k_b = (4*pi)/(3*a)* [0,1]
r_OF = 2/3*v_a + 1/3*v_b
r_OD = -1/2*v_a + 1/3*v_b
r_OG = -1/3*v_a - 2/3*v_b

ks(m,n) = m/Lm*k_a + n/Ln * k_b
Eu(x) = cos(x)+im*sin(x)
Δ(k) = Eu(dot(k,r_OF)) + Eu(dot(k,r_OD))+Eu(dot(k,r_OG))
h(k) = [[mass,conj(Δ(k))] [Δ(k),-mass]]
invG(ω,k) = im*ω*I -h(k)
G(ω,k) = inv(invG(ω,k))


correlator = zeros(ComplexF64,(Nt,2,2))
x = [0,0]
y = [0,0]
ω(t) = pi*(2t+1)/β
for τ=1:Nt
    for m=1:Lm
        for n=1:Ln
            for t = -40:40
                correlator[τ,:,:] .+=G(ω(t),ks(m,n)).*Eu(dot(ks(m,n),(x-y))).*Eu(ω(t)*(τ-1)*(β/Nt))
            end 
        end 
    end 
end 
@show size(correlator)

correlator ./=Lm*Ln*β
@show size(correlator)
@show real(correlator[:,1,1])

τ = (0:1:Nt-1).*(β/Nt)
clf()
plot(τ, real(correlator[:,1,1]),".")
xlabel(L"time")
ylabel(L"\langle G_AA(τ,x,y) \rangle")
savefig("GAA")

clf()
plot(τ, real(correlator[:,1,2]),".")
xlabel(L"time")
ylabel(L"\langle G_AB(τ,x,y) \rangle")
savefig("GAB")

clf()
plot(τ, real(correlator[:,2,1]),".")
xlabel(L"time")
ylabel(L"\langle G_BA(τ,x,y) \rangle")
savefig("GBA")

clf()
plot(τ, real(correlator[:,2,2]),".")
xlabel(L"time")
ylabel(L"\langle G_BB(τ,x,y) \rangle")
savefig("GBB")






