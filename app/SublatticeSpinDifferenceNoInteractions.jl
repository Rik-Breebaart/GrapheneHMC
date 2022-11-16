#=This file will contain the computations of the greens function of the non-interacting graphene tight binding model.

The correlator will be computed in two ways. using the Fermionic matrix and using the analytical summation. 
We will also check both in k-space and position space keeping the temporal component the same.
=#

using Distributions, Random, Random.DSFMT, LinearAlgebra, PyPlot, CurveFit
include(abspath(@__DIR__, "../src/analyticalNoInteractions.jl"))
include(abspath(@__DIR__, "../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/hamiltonianNoInteractions.jl"))
include(abspath(@__DIR__, "../src/observables.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))


lat = Lattice(4, 4, 16)

# ms = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
ms = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

Δn_array = zeros(length(ms)[1])
Δn_analytical = zeros((length(ms)[1]))
par_0 = Parameters(2.0, 0.0, 1.0, 0.5)
for i in 1:length(ms)[1]
    par = Parameters(par_0.β, ms[i], 1.0, 0.5)
    M_no_int = FermionicMatrix_no_int(par, lat)
    Δn_analytical[i] = real.(Δn_no_int(par, lat))
    Δn_array[i] = Δn(M_no_int, par, lat)
end
m_x = 0.0:0.01:0.6

int_ms = [1,2,3,4,5]
clf()
plot(ms, Δn_array, ".", label=L"$\alpha = 0.0$")
fit = curve_fit(Polynomial, ms[int_ms], Δn_array[int_ms], 2)
y0b = fit.(m_x) 
plot(m_x,y0b,"-",label="fit")
plot(m_x, m_x.*(1/2), label="trial")
# plot(ms, Δn_analytical, ".", label=L"analytical")
legend()
title(L"$\Delta n$ no interactions")
xlabel("m")
ylabel("Δn")
savefig(abspath(@__DIR__,string("../plots/SublatticeSpin_no_int_",lat.Lm,"_",lat.Ln,"_",lat.Nt,".png")))

Δn_array = zeros((length(ms)[1],lat.Nt))
Δn_analytical = zeros((length(ms)[1],lat.Nt))
for i in 1:length(ms)[1]
    par = Parameters(par_0.β, ms[i], 1.0, 0.5)
    M_no_int = FermionicMatrix_no_int(par, lat)
    Δn_analytical[i,:] = real.(Δn_no_int_time(par, lat))
    Δn_array[i,:] = Δn_time(M_no_int, par, lat)
end

# τ = (0:1:lat.Nt-1).*(par_0.β/lat.Nt)
# for i in 1:length(ms)[1]
#     clf()
#     plot(τ, Δn_analytical[i,:], "*", label=string("analytical m = ",ms[i]))
#     plot(τ, Δn_array[i,:], ".", label=string("M: m = ",ms[i]))
#     legend()
#     title(L"$\Delta n$ no interactions")
#     xlabel("τ")
#     ylabel("Δn")
#     savefig(abspath(@__DIR__,string("../plots/SublatticeSpin_no_int_",lat.Lm,"_",lat.Ln,"_",lat.Nt,"_time.png")))
# end

# Nts = [4, 8, 12, 16, 20, 24]
# Δn_array = zeros(length(ms)[1],length(Nts)[1])
# par_0 = Parameters(2.0, 0.0, 1.0, 0.5)
# for i in 1:length(ms)[1]
#     par = Parameters(par_0.β, ms[i], 1.0, 0.5)
#     for j in 1:length(Nts)[1]
#         change_lat(lat, Nt=Nts[i])
#         M_no_int = FermionicMatrix_no_int(par, lat)
#         Δn_array[i,j] = Δn(M_no_int, par, lat)
#     end 
# end

# clf()
# Nts_x = 0.0:0.01:((1/4)+0.05)
# for i in 1:length(ms)[1]
#     plot(1 ./Nts, Δn_array[i,:], ".")
#     fit = curve_fit(Polynomial, 1 ./Nts, Δn_array[i,:], 1)
#     y0b = fit.(Nts_x) 
#     plot(Nts_x, y0b, "--", linewidth=1)
# end 
# xlabel("1/Nt")
# ylabel(L"$\langle \Delta n \rangle$")
# title(string(L"$\beta$ = ", par_0.β ,L" $\alpha $= ",0.0))
# xlim([0.0, (1/8)+0.05])
# grid()
# savefig(abspath(@__DIR__,string("../plots/SublatticeSpin_hmc_thermilization_non_interacting_",lat.Lm,"_",lat.Ln,"_",lat.Nt,".png")))
