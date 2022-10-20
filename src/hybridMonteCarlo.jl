#= 
This file will contain the implementation for hybrid monte carlo in a general manner
    =# 

using Random, Random.DSFMT, Distributions, LinearAlgebra
include("integrators.jl")

mutable struct HMC_Parameters
    Nsamples::Integer
    path_length::Float64
    step_size::Float64
    offset::Integer
    m_sw::Integer
    burn_in::Integer
    equation::Integer
end 

function HMC_LeapFrog_default(Nsamples)
    offset = floor(Integer, 0.3*Nsamples)
    if Nsamples <400
        burn_in = floor(Integer, 0.25*Nsamples)
    else 
        burn_in = 100
    end
    return HMC_Parameters(Nsamples, 10.0, 0.05, offset, 1, burn_in)
end

function HMC_SW_default(Nsamples; step_size=0.5, m_sw = 5, path_length=10.0)
    offset = floor(Integer, 0.3*Nsamples)
    if Nsamples <400
        burn_in = floor(Integer, 0.25*Nsamples)
    else 
        burn_in = 100
    end
    return HMC_Parameters(Nsamples, path_length, step_size, offset, m_sw, burn_in)
end



"""
Hybrid monte carlo following the code from https://gist.github.com/Shoichiro-Tsutsui/53eb534f6794e1eece55b1d5a7c118fe 
with the fields changed to reflect those in the paper (10.1103/PhysRevB.89.195429)
The hybrid monte caro performs the following steps:
step 0: initialize the system
step 1: sample the momentum 
step 2: Perform the hamiltonian monte carlo by using an integrator
step 3: compare the difference between the hamiltonian before and after the integration scheme and keep based on metropolis acceptance

Input: 
    S       (function)      The function corresponding to the action of the system
    ∇S      (function)      The funtion corresponding to the derivative of the action
    M_funciton (function)   Function which creates the fermionic matrix
    D       (Integer)       The dimensions of the field (space X time)
    path_length (Float)     The path length of the integration scheme
    step_size  (Float)      The size of the steps in the integration
    Nsample (Integer)       The number of samples obtained using the montecarlo scheme (all the accepted cases)
Optional:
    rng                     The Random number generator used (if not provided MersenneTwister() is used)
    position_init  (Vector of D Floats or single float)         The initial value of the position field from which the hamiltonian dynamics starts (default: 1.0)
    print_time     (bool)   Indicates wheter the run time of single sample is printed (default=true)
    print_accept   (bool)   Indicates wheter the acceptance of single sample is printed (default=true)      
    print_H=false  (bool)   Indicates whether the hamiltonian difference is printed after each integration (default=false)
Output:
    conf    (Nsample X n float) The obtained Nsample configurations of the discretized field
    nreject  (Integer)      The nunber of rejections before reaching the Nsample configurations
"""       
function HybridMonteCarlo(S::Function, ∇S::Function, D::Integer, path_length, step_size, Nsamples::Integer; rng=MersenneTwister(), position_init=1.0, print_time=true, print_accept=true, print_H=false)
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


"""
Metropolis Hastings Monte Carlo sampler

Input: 
    S      (function)      The function corresponding to the action of the system
    D       (Integer)       The dimensions of the field
    step_size  (Float)      The size of the steps in the integration
    Nsample (Integer)       The number of samples obtained using the montecarlo scheme (all the accepted cases)
Optional:
    rng                     The Random number generator used (if not provided MersenneTwister() is used)
    position_init  (Vector of D Floats or single float)         The initial value of the position field from which the hamiltonian dynamics starts (default: 1.0)
    print_time     (bool)   Indicates wheter the run time of single sample is printed (default=true)
    print_accept   (bool)   Indicates wheter the acceptance of single sample is printed (default=true)      
    print_H=false  (bool)   Indicates whether the hamiltonian difference is printed after each integration (default=false)  
Output:
    conf    (Nsample X n float) The obtained Nsample configurations of the discretized field
    nreject  (Integer)      The nunber of rejections before reaching the Nsample configurations
"""
function MetropolisHastingsMonteCarlo(S, D, step_size, Nsamples; rng=MersenneTwister(), position_init=1.0, print_time=false, print_accept=false, print_H=false)
    #set up empty memory for the position and 
    configurations = zeros(Nsamples,D)

    #count the rejections
    nreject = 0
    if size(position_init)[1]>1
        position = position_init
    elseif size(position_init)[1] == 1
        position = (2*rand(rng,D).-1).*position_init
    else 
        error("incorrect initial position is given")
    end 
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
        if print_H===true
            @show ΔH
        end
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


#========{now hybrid monte carlo specific for the graphene problem (thus also including a complex gaussian ρ)}==========#

"""
Metropolis Hatings monte carlo for graphene
with the fields changed to reflect those in the paper (10.1103/PhysRevB.89.195429)
The hybrid monte caro performs the following steps:
step 0: initialize the system
step 1: sample the momentum 
step 2: Perform the hamiltonian monte carlo by using an integrator
step 3: compare the difference between the hamiltonian before and after the integration scheme and keep based on metropolis acceptance

Input: 
    S       (function)      The function corresponding to the action of the system
    ∇S      (function)      The funtion corresponding to the derivative of the action
    M_funciton (function)   Function which creates the fermionic matrix
    D       (Integer)       The dimensions of the field (space X time)
    path_length (Float)     The path length of the integration scheme
    step_size  (Float)      The size of the steps in the integration
    Nsample (Integer)       The number of samples obtained using the montecarlo scheme (all the accepted cases)
Optional:
    rng                     The Random number generator used (if not provided MersenneTwister() is used)
    position_init  (Vector of D Floats or single float)         The initial value of the position field from which the hamiltonian dynamics starts (default: 1.0)
    print_time     (bool)   Indicates wheter the run time of single sample is printed (default=true)
    print_accept   (bool)   Indicates wheter the acceptance of single sample is printed (default=true)      
    print_H=false  (bool)   Indicates whether the hamiltonian difference is printed after each integration (default=false)
Output:
    conf    (Nsample X n float) The obtained Nsample configurations of the discretized field
    nreject  (Integer)      The nunber of rejections before reaching the Nsample configurations
"""
function HybridMonteCarlo(S::Function, ∇S::Function, M_function::Function, D::Integer, path_length, step_size,
                        Nsamples::Integer; rng=MersenneTwister(), position_init=1.0, 
                        print_time=true, print_accept=true, print_H=false, burn_in = 0)
    #set up empty memory for the position and 
    if length(position_init)[1] == 1
        ϕ = (2*rand(rng,D).-1).*position_init
    elseif length(position_init)[1]>1
        ϕ = position_init
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
            @show real(ΔH)
        end
        #step 3B: perform metropolis check
        if i > burn_in
            if randU[i] > exp(-real(ΔH)) 
                nreject += 1
                ϕ = ϕ_old
            else
                ϕ = ϕ_trial
                if print_accept==true
                    println("Accepted ",i-nreject-burn_in)
                end
            end
        else
            if abs(real(ΔH)) <100
                ϕ = ϕ_trial
                println("Accepted burn in",i-nreject-burn_in)
            else 
                nreject +=1
                ϕ = ϕ_old
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

"""
Metropolis Hatings monte carlo for graphene
with the fields changed to reflect those in the paper (10.1103/PhysRevB.89.195429)
The hybrid monte caro performs the following steps:
step 0: initialize the system
step 1: sample the momentum 
step 2: Perform the hamiltonian monte carlo by using an integrator
step 3: compare the difference between the hamiltonian before and after the integration scheme and keep based on metropolis acceptance

Input: 
    S       (function)      The function corresponding to the action of the system
    ∇S      (function)      The funtion corresponding to the derivative of the action
    M_funciton (function)   Function which creates the fermionic matrix
    D       (Integer)       The dimensions of the field (space X time)
    path_length (Float)     The path length of the integration scheme
    step_size  (Float)      The size of the steps in the integration
    Nsample (Integer)       The number of samples obtained using the montecarlo scheme (all the accepted cases)
Optional:
    rng                     The Random number generator used (if not provided MersenneTwister() is used)
    position_init  (Vector of D Floats or single float)         The initial value of the position field from which the hamiltonian dynamics starts (default: 1.0)
    print_time     (bool)   Indicates wheter the run time of single sample is printed (default=true)
    print_accept   (bool)   Indicates wheter the acceptance of single sample is printed (default=true)      
    print_H=false  (bool)   Indicates whether the hamiltonian difference is printed after each integration (default=false)
Output:
    conf    (Nsample X n float) The obtained Nsample configurations of the discretized field
    nreject  (Integer)      The nunber of rejections before reaching the Nsample configurations
"""
function HybridMonteCarlo(S::Function, ∇S_V::Function, ∇S_M::Function, M_function::Function, D::Integer, path_length, step_size, m::Integer, 
                        Nsamples::Integer; rng=MersenneTwister(), position_init=1.0, 
                        print_time=true, print_accept=true, print_H=false, burn_in = 0)
    #set up empty memory for the position and 
    if length(position_init)[1] == 1
        ϕ = (2*rand(rng,D).-1).*position_init
    elseif length(position_init)[1]>1
        ϕ = position_init
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
        dqdt_V(ϕ) = -∇S_V(ϕ, χ)
        dqdt_M(ϕ) = -∇S_M(ϕ, χ)
        #step 2A: compute the original hamiltonian energy
        H_init = 0.5*sum(π.^2) + S(ϕ, χ)
        #step 2B: perform the molecular dynamics using the prescribed integrator
        if i<burn_in
            ϕ_trial, π = SextonWeingartenIntegrator(path_length, step_size/5, ϕ, π, dqdt_V, dqdt_M, m)
        else 
            ϕ_trial, π = SextonWeingartenIntegrator(path_length, step_size, ϕ, π, dqdt_V, dqdt_M, m)
        end
        #step 2C: compute the final hamiltonian energy
        H_final = 0.5*sum(π.^2) + S(ϕ_trial, χ)

        #step 3A: compute the energy difference for Metropolis-Hastings check
        ΔH = H_final-H_init
        if print_H==true
            @show real(ΔH)
        end
        #step 3B: perform metropolis check
        if i > burn_in
            if randU[i] > exp(-real(ΔH)) 
                nreject += 1
                ϕ = ϕ_old
            else
                ϕ = ϕ_trial
                if print_accept==true
                    println("Accepted ",i-nreject)
                end
            end
        else
            if abs(real(ΔH)) <100
                ϕ = ϕ_trial
                println("Accepted burn in ",i-nreject)
            else 
                nreject +=1
                ϕ = ϕ_old
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
