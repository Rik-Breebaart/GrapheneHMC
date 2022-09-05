
using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot
include(abspath(@__DIR__,"../src/analyticalNoInteractions.jl"))


lat = Lattice(6,6,8,1.0)
par = Parameters(2.0, 0.0, 1.0, 0.5)
int(x) = floor(Int, x)

x=[0,0]
y=[0,0]
correlator = greensFunctionGraphene(x,y,par,lat)

τ = (0:1:lat.Nt-1).*(par.β/lat.Nt)
clf()
plot(τ, real(correlator[:,1,1]),".")
xlabel(L"time")
ylabel(L"\langle G_{AA}(τ,x,y) \rangle")
savefig("/home/rikbre/GrapheneHMC/plots/GAA")

clf()
plot(τ, real(correlator[:,1,2]),".")
xlabel(L"time")
ylabel(L"\langle G_{AB}(τ,x,y) \rangle")
savefig("/home/rikbre/GrapheneHMC/plots/GAB")

clf()
plot(τ, real(correlator[:,2,1]),".")
xlabel(L"time")
ylabel(L"\langle G_{BA}(τ,x,y) \rangle")
savefig("/home/rikbre/GrapheneHMC/plots/GBA")

clf()
plot(τ, real(correlator[:,2,2]),".")
xlabel(L"time")
ylabel(L"\langle G_{BB}(τ,x,y) \rangle")
savefig("/home/rikbre/GrapheneHMC/plots/GBB")






