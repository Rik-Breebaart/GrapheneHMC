using Random, Distributions, PyPlot, Test

function Test_complexNormal()
    D = 10
    N_samples = 100000
    rng = MersenneTwister(123)
    d = Normal(0,1/sqrt(2))
    @time ρ = rand(d, (D,N_samples)) + im*rand(d, (D,N_samples))
    # ρ = x[1, :, :] + im*x[2, :, :]
    @time ρ_rand = randn(rng, ComplexF64, (D, N_samples))
    
    clf()
    # scatter(real(ρ[1,:]),imag(ρ[1,:]),label="rho")
    scatter(real(ρ_rand[1,:]),imag(ρ_rand[1,:]),label="rhorng")
    scatter(real(ρ[1,:]),imag(ρ[1,:]),label="rho")
   
    legend()
    savefig(abspath(@__DIR__,"../plots/Test_complexNormal.png"))
end 

Test_complexNormal()
    

