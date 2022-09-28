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
    steps = Integer(ceil(path_len/step_size))
    
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
    H   (path_len/step_size array) The total energy of the system throughout the integration
    U   (path_len/step_size array) The potential energy of the sytem throughout the integration
    K   (path_len/step_size array) The Kinetic energy of the sytem throughout the integration
"""
function LeapFrogPQP_store(path_len, step_size, p, q, dqdt, pot)
    # the input are the initial coordinates p0 and q0 and the functions dpdt and dqdt for the derivatives of the LeapFrog
    
    # perform LeapFrog for N=path_len/step_size steps
    steps = Integer(ceil(path_len/step_size))

    H = zeros(ComplexF64,(steps))
    K = zeros(ComplexF64,(steps))
    U = zeros(ComplexF64,(steps))
    #using QPQ scheme
    K[1] = 0.5*sum(q.*q)
    U[1]= pot(p)
    H[1] = K[1]+ U[1]

    for s = 2:steps
        p += step_size*q/2
        q += step_size*dqdt(p)
        p += step_size*q/2
        K[s] = 0.5*sum(q.*q)
        U[s]= pot(p)
        H[s] = K[s]+ U[s]
    end
    return p, q, H, K, U
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
    H   (path_len/step_size array) The total energy of the system throughout the integration
    U   (path_len/step_size array) The potential energy of the sytem throughout the integration
    K   (path_len/step_size array) The Kinetic energy of the sytem throughout the integration
"""
function LeapFrogQPQ_store(path_len, step_size, p, q, dqdt, pot)
    # the input are the initial coordinates p0 and q0 and the functions dpdt and dqdt for the derivatives of the LeapFrog
    
    # perform LeapFrog for N=path_len/step_size steps
    steps = Integer(ceil(path_len/step_size))

    H = zeros(ComplexF64,(steps))
    K = zeros(ComplexF64,(steps))
    U = zeros(ComplexF64,(steps))
    #using QPQ scheme
    K[1] = 0.5*sum(q.*q)
    U[1]= pot(p)
    H[1] = K[1]+ U[1]
    a = dqdt(p) 
    for s = 2:steps
        q += step_size*a/2 
        p += step_size*q
        a = dqdt(p) 
        q += step_size*a/2
        K[s] = 0.5*sum(q.*q)
        U[s]= pot(p)
        H[s] = K[s]+ U[s]
    end
    return p, q, H, K, U
end

function SextonWeingartenIntegrator(path_len, step_size, p, q, dqdt_slow, dqdt_fast, n)
        # the input are the initial coordinates array p0 and q0 and the functions dpdt and dqdt for the derivatives of the LeapFrog
    
    #perform LeapFrog for N=path_len/step_size steps
    steps = Integer(ceil(path_len/step_size))
    #using QPQ scheme
    a1 = dqdt_slow(p) 
    a2 = dqdt_fast(p)
    q += step_size*a1/2 
    for s = 2:steps
        q += step_size*a2/(2*n) 
        for i = 1:(n-1)
            p += step_size*q/n
            a2 = dqdt_fast(p)
            q += step_size*a2/(n) 
        end
        p += step_size*q/n
        a2 = dqdt_fast(p)
        a1 = dqdt_slow(p) 
        q += step_size*a2/(2*n) 
        if s === steps       
            q += step_size*a1/2
        else 
            q += step_size*a1
        end
    end
    return p, q
end


function SextonWeingartenIntegrator_store(path_len, step_size, p, q, dqdt_slow, dqdt_fast, pot, n)
    #perform integration for N=path_len/step_size steps
    steps = Integer(ceil(path_len/step_size))
        
    H = zeros(ComplexF64,(steps))
    K = zeros(ComplexF64,(steps))
    U = zeros(ComplexF64,(steps))
    #using QPQ scheme
    K[1] = 0.5*sum(q.*q)
    U[1]= pot(p)
    H[1] = K[1]+ U[1]
    a1 = dqdt_slow(p) 
    a2 = dqdt_fast(p)
    for s = 2:steps
        q += step_size*a1/2 
        for i = 1:n
            q += step_size*a2/(2*n) 
            p += step_size*q/n
            a2 = dqdt_fast(p)
            q += step_size*a2/(2*n) 
        end
        a1 = dqdt_slow(p) 
        q += step_size*a1/2
        K[s] = 0.5*sum(q.*q)
        U[s]= pot(p)
        H[s] = K[s]+ U[s]
    end
    return p, q, H, K, U
end