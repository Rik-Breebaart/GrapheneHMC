#= 
This file will contain the integrators that can be used for integration in the hybrid monte carlo molecular dynamics component.

=#

"""
Leap frog integrations scheme using position momentum position (PQP) scheme.

Input: 
    path_len (Float64)      The path length for the number of steps performed
    step_size (Float64)     The step size for each integration step
    p_0     (Float Vector)  The initial positions
    q_0     (Float Vector)  The initial momentum
    dpdt    (Function)      The position time derivative (function of p and q)
    dqdt    (Function)      The momentum time derivative (function of p and q)

Out:    
    P   (Float vector)      The final position vector
    q   (Float vector)      The final momentum vector
"""
function LeapFrogPQP(path_len, step_size, p_0, q_0, dpdt, dqdt)
    # the input are the initial coordinates array p0 and q0 and the functions dpdt and dqdt for the derivatives of the LeapFrog
    
    #perform LeapFrog for N=path_len/step_size steps
    steps = path_len/step_size
    
    p = p_0
    q = q_0
    #using PQP scheme
    p .+= step_size.*dpdt.(q)/2 
    for s = 1:steps
        q .+= step_size.*dqdt.(p)
        p .+= step_size.*dpdt.(q)
    end
    q .+= step_size.*dqdt.(p)
    p .+= step_size.*dpdt.(q)/2 
    return p, q
end

"""
Leap frog integrations scheme using momentum position momentum(QPQ) scheme.

Input: 
    path_len (Float64)      The path length for the number of steps performed
    step_size (Float64)     The step size for each integration step
    p_0     (Float Vector)  The initial positions
    q_0     (Float Vector)  The initial momentum
    dpdt    (Function)      The position time derivative (function of p and q)
    dqdt    (Function)      The momentum time derivative (function of p and q)

Out:    
    P   (Float vector)      The final position vector
    q   (Float vector)      The final momentum vector
"""
function LeapFrogQPQ(path_len, step_size, p_0, q_0, dpdt, dqdt)
    # the input are the initial coordinates p0 and q0 and the functions dpdt and dqdt for the derivatives of the LeapFrog
    
    #perform LeapFrog for N=path_len/step_size steps
    steps = path_len/step_size
    
    p = p_0
    q = q_0
    #using QPQ scheme
    q += step_size*dqdt(p)/2 
    for s = 1:steps-1
        p += step_size*dpdt(q)
        q += step_size*dqdt(p)
    end
    p += step_size*dpdt(q)
    q += step_size*dqdt(p)/2
    return p, q
end
