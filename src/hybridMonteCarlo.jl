#= 
This file will contain the implementation for hybrid monte carlo in a general manner
    =# 

using Random, Random.DSFMT, Distributions, LinearAlgebra
include("integrators.jl")

        
function HybridMonteCarlo(S::Function, ∇S::Function, D::Integer, path_length, step_size, Nsamples::Integer; rng=MersenneTwister(), position_init=1.0, print_time=true, print_accept=true)
    #set up empty memory for the position and 
    if size(position_init)[1]>1
        position = position_init
    elseif size(position_init)[1] == 1
        position = (2*rand(rng,D).-1).*position_init
    else 
        error("incorrect initial position is given")
    end 
    configurations = zeros(Nsamples,D)
    randU = rand(rng,Float64,(Nsamples))

    #count the rejections
    nreject = 0
    dqdt(p) = -∇S(p)
    dpdt(q) = q
    
    #compute Nsamples different configurations
    for i =1:Nsamples
        if  print_time==true
            println("start ",i)
            starttime = time()
        end
        #step 1: Compute random momentum from gaussian distribution with weight momentum_mass
        momentum = rand(rng, Normal(),D)
        position_old = copy(position)
        #step 2A: compute the original hamiltonian energy
        H_init = 0.5*sum(momentum.^2) + S(position)
        #step 2B: perform the molecular dynamics using the prescribed integrator
        position_trial, momentum = LeapFrogQPQ(path_length, step_size, position, momentum, dpdt, dqdt)
        #step 2C: compute the final hamiltonian energy
        H_final = 0.5*sum(momentum.^2) + S(position_trial)

        #step 3A: compute the energy difference for Metropolis-Hastings check
        ΔH = H_final-H_init
        #step 3B: perform metropolis check
        if randU[i] > exp(-real(ΔH))
            nreject += 1
            position = position_old
        else
            position = position_trial
            if print_accept==true
                println("Accepted ",i-nreject)
            end
        end
        configurations[i,:] .= position

        if  print_time==true            
            endtime = time()
            println("end! time: ",endtime-starttime)
        end
    end
    # return the final configurations and the number of rejections.
    return configurations, nreject
end 


function MetropolisHastingsMonteCarlo(S, D, step_size, Nsamples; rng=MersenneTwister(), position_init=1.0, print_time=false, print_accept=false)
    #set up empty memory for the position and 
    configurations = zeros(Nsamples,D)

    #count the rejections
    nreject = 0
    position = (2*rand(rng,D).-1).*position_init
    position_trial = zeros(D)
    randstep = rand(rng,Float64, (Nsamples, D)).*2 .-1
    randU = rand(rng,Float64,(Nsamples))
    #compute Nsamples different configurations
    for i =1:Nsamples
        if  print_time==true
            println("start ",i)
            starttime = time()
        end

        #step 2A: compute the original hamiltonian energy
        H_init = S(position)
        #step 2B: perform the random step
        position_trial = position + step_size.*randstep[i,:]
        #step 2C: compute the final hamiltonian energy
        H_final = S(position_trial)

        #step 3A: compute the energy difference for Metropolis-Hastings check
        ΔH = H_final-H_init
        #step 3B: perform metropolis check
        if randU[i] >= exp(-real(ΔH))
            nreject += 1
        else
            position = position_trial
            if print_accept==true
                println("Accepted ",i)
            end
        end
        configurations[i,:] .= position

        if  print_time==true            
            endtime = time()
            println("end! time: ",endtime-starttime)
        end
    end
    # return the final configurations and the number of rejections.
    return configurations, nreject
end 


#now hybrid monte carlo specific for the graphene problem (thus also including a complex gaussian ρ)


function HybridMonteCarlo(S::Function, ∇S::Function, M_function::Function, D::Integer, path_length, step_size, Nsamples::Integer; rng=MersenneTwister(), position_init=1.0, print_time=true, print_accept=true, print_H=false)
    #set up empty memory for the position and 
    if size(position_init)[1]>1
        ϕ = position_init
    elseif size(position_init)[1] == 1
        ϕ = (2*rand(rng,D).-1).*position_init
    else 
        error("incorrect initial position is given")
    end 
    configurations = zeros(Nsamples,D)
    randU = rand(rng,Float64,(Nsamples))
    randρ = randn(rng,ComplexF64, (Nsamples,D))

    #count the rejections
    nreject = 0
    dpdt(π) = π
    #compute Nsamples different configurations
    for i =1:Nsamples
        if  print_time==true
            println("start ",i)
            starttime = time()
        end
        #step 1: Compute random momentum from gaussian distribution with weight momentum_mass
        π = rand(rng, Normal(),D)
        ϕ_old = copy(ϕ)
        M = M_function(ϕ)
        χ = M*randρ[i,:]
        dqdt(ϕ) = -∇S(ϕ, χ)
        #step 2A: compute the original hamiltonian energy
        H_init = 0.5*sum(π.^2) + S(ϕ, χ)
        #step 2B: perform the molecular dynamics using the prescribed integrator
        ϕ_trial, π = LeapFrogQPQ(path_length, step_size, ϕ, π, dpdt, dqdt)
        #step 2C: compute the final hamiltonian energy
        H_final = 0.5*sum(π.^2) + S(ϕ_trial, χ)

        #step 3A: compute the energy difference for Metropolis-Hastings check
        ΔH = H_final-H_init
        if print_H==true
            @show ΔH
        end
        #step 3B: perform metropolis check
        if randU[i] > exp(-real(ΔH))
            nreject += 1
            ϕ = ϕ_old
        else
            ϕ = ϕ_trial
            if print_accept==true
                println("Accepted ",i-nreject)
            end
        end
        configurations[i,:] = ϕ

        if  print_time==true            
            endtime = time()
            println("end! time: ",endtime-starttime)
        end
    end
    # return the final configurations and the number of rejections.
    return configurations, nreject
end 

