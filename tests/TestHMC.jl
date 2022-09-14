#= 
This file contains the tests for the HybridMonteCarlo implementation
=#
using PyPlot, Test
include(abspath(@__DIR__,"../src/hybridMonteCarlo.jl"))
include(abspath(@__DIR__,"../src/integrators.jl"))

function Test_HybirdMonteCarlo()
    # Gaussian integral
    S(x) = sum(x.^2/2)
    O(x) = mean(x)
    ∇S(x) = x

    Nsamples = 10000
    D =1
    path_length = 5.0
    step_size = 0.1
    configuration, rejections = HybridMonteCarlo(S, ∇S, D, path_length, step_size, Nsamples, print_time=false, print_accept=false)
    res = [O(configuration[i, :]) for i in 1:Nsamples]
    @show rejections
    clf()
    plot(res)
    savefig(abspath(@__DIR__,"../plots/Test_HMC_gaussian.png"))
    @test isapprox(mean(res),0.0,atol=0.05)
    @test isapprox(std(res),1.0,atol=0.05)
end 


function Test_MetropolisHastingsMonteCarlo()
        # Gaussian integral
        S(x) = sum(x.^2/2)
        O(x) = mean(x)
    
        Nsamples = 1000000
        interval = 1000
        D =1
        step_size = 0.01
        configuration, rejections = MetropolisHastingsMonteCarlo(S, D, step_size, Nsamples, print_time=false, print_accept=false)
        res = [O(configuration[i*interval, :]) for i in 1:floor(Int,Nsamples/interval)]
        @test isapprox(mean(res),0.0,atol=0.1)
        @test isapprox(std(res),1.0,atol=0.1)
end 

Test_HybirdMonteCarlo()
# Test_MetropolisHastingsMonteCarlo()
