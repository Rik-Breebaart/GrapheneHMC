#=This file will contain the functions and struct for the hexagonal lattice found in graphene.=#


"""
Lattice struct containing the lattice paramaters

Arguments:
    Lm (int)        The m length of the graphene lattice grid (horizontal length)
    Ln (int)        The n length of the graphene lattice grid (vertical length)
    Nt (int)        The number time steps 
    a  (Float)      The lattice spacing
    dim_sub (Int)   The number of sites per sublattice (Lm*Ln)
    D  (int)        The total dimension of the field (Lm*Ln*Nt*2)
"""
struct Lattice
    Lm::Int
    Ln::Int
    Nt::Int
    a::Float64
    dim_sub::Int
    D::Int
end 

# automatically compute the lattice dimension when the struct is created.
Lattice(Lm,Ln,Nt) = Lattice(Lm,Ln,Nt,0.71*10^(-3))
Lattice(Lm,Ln,Nt,a) = Lattice(Lm,Ln,Nt,a,Lm*Ln)
Lattice(Lm,Ln,Nt,a,dim_sub) = Lattice(Lm,Ln,Nt,a,dim_sub,dim_sub*2*Nt)

struct Particle
    m::Int
    n::Int
    Pab::Int
    start::Int
end 

Particle(m, n, Pab) = Particle(m, n, Pab, 0) 

function particle_position(particle::Particle, lat::Lattice)
    return particle_position(particle.m, particle.n, particle.Pab, lat, start=particle.start)
end 

function particle_position(m, n, Pab, lat::Lattice; start=0)
    bas1 = [sqrt(3)/2,0]
    bas2 = [0,1/2]
    return ((2*(m-start)+(n-start))*bas1 + (3*(n-start)+2*Pab)*bas2).*lat.a
end 


"""
The index of each field point on the lattice including time evolution of Nt steps. 
The first Lm*Ln*Nt are on the A sublatice and the second are on the B sublatice.
This function uses the indexing starting at 0 for m,n and τ, it outputs a index starting at 1

Input:  
    m (int)                 the m index
    n (int)                 the n index
    τ (int)                 the time index
    Pab (Bool)              The boolean indicator on which lattice we are if Pab=0 then sublatice A elseif Pab=1 on sublatice B
    lat (Lattice struct)    The lattice struct contianing the lattice parameters

Output:
    index (int)             The index of the field site of the graphene lattice
"""
function index(m, n, τ, Pab, lat;start=0)
     #m,n and τ index starting at 0
    if start==0
        if m<start || n<start || τ<start
            error("incorrect m,n, or τ given. should be larger then start")
        end
        return m + n*lat.Lm + τ*(lat.Lm*lat.Ln) + Pab*(lat.Lm*lat.Ln*lat.Nt) + 1 
    elseif start==1
        index(m-1, n-1, τ-1, Pab, lat,start=0)
    end 
end 

"""
The index of each field point on the lattice when no time evolution occurs.
The first Lm*Ln*Nt are on the A sublatice and the second are on the B sublatice.
This function uses the indexing starting at 0 for m,and n, it outputs a index starting at 1

Input:  
    m (int)                 the m index
    n (int)                 the n index
    Pab (Bool)              The boolean indicator on which lattice we are if Pab=0 then sublatice A elseif Pab=1 on sublatice B
    lat (Lattice struct)    The lattice struct contianing the lattice parameters

Output:
    index (int)             The index of the field site of the graphene lattice
"""
function index(m,n,Pab,lat;start=0)
    #m,n and index starting at 1 (equal time index (Nt=1))
    #if no Nt is given the index function for Nt=1 is used.
    return (m-start) + (n-start)*lat.Lm + Pab*(lat.Lm*lat.Ln) + 1
end


""" Function to create neighbour matrix

Input: 
    lat (Lattice struct)    Struct containing the lattice paramaters

Output:
    M   (2*Lm*Ln x 2*Lm*Ln matrix)    Boolean matrix labeling the coordinates of the neighbours
"""
function neighbour_matrix(lat::Lattice) 
    A = zeros((lat.dim_sub*2,lat.dim_sub*2))
    Pab = 0
    for m=1:lat.Lm, n=1:lat.Ln
        neighbours = neighbours_mnIndex(m, n, Pab, lat, start=1)
        #the neighbours with the lattice on which they live (eg. Pab_n = |Pab-1|)
        for i = 1:1:3
            (m_n,n_n,Pab_n) = neighbours[i,:]
            # add the neighbour interactions 
            A[index(m, n, Pab, lat, start=1), index(m_n, n_n, Pab_n, lat, start=1)] =+1
        end
    end
    return A + A'
end 


function neighbours_x(x, Pab, lat::Lattice)
    bas1 = [1,0]
    bas2 = [0,1]
    Neighbours = zeros(3,3)
    Neighbours[:,3] .= abs(Pab-1)
    if Pab==1
        q = -1
    else 
        q = 1 
    end
    Neighbours[1,1:2] = x + q*2*bas2
    Neighbours[2,1:2] = x + q*(bas1-bas2)
    Neighbours[3,1:2] = x + q*(-bas1 - bas2)
    return Neighbours
end 


"""
Function returning the indexes corresponging to the neighbours of the input site,with m,n index starting from 0

Input:
    m   (int)       The m index of the input site
    n   (int)       The n index of the input site
    Pab (boolean)   The boolean indicating the sublatice on which the input site lives
    lat (Lattice)   The lattice struct containing the lattice Parameters

Output:
    Neighbours_mn (3x3 array of integers)  Array containing the m,n indices of the neighbours of the input site (the [:,3] labels the sublattice)
"""
function neighbours_mnIndex(m, n, Pab, lat::Lattice; start=0)
    #= this function returns the coordinates of the neighbours of x (a 2d vector)
    =#
    m_val(x) = x[1]/2 - x[2]/6 + 1/3*x[3]
    n_val(x) = (x[2] - 2*x[3])/3
    x= [2*(m-start)+(n-start), 3*(n-start)+2*Pab]
    Neighbours = neighbours_x(x,Pab,lat)
    Neighbours_mn = zeros(Int,(3,3))
    for i = 1:1:3
        #loop over the three neighbours of the hexagonal lattice.
        Neighbours_mn[i,:] = [round(m_val(Neighbours[i,:])),round(n_val(Neighbours[i,:])), Neighbours[i,3]]
        if Neighbours_mn[i,2] < 0
            Neighbours_mn[i,2] += lat.Ln
            Neighbours_mn[i,1] -= round(Int,lat.Ln/2)
        end
        if Neighbours_mn[i,2] >= lat.Ln
            Neighbours_mn[i,2] -= lat.Ln
            Neighbours_mn[i,1] += round(Int,lat.Ln/2)
        end
        if Neighbours_mn[i,1] < 0
            Neighbours_mn[i,1] += lat.Lm
        end
        if Neighbours_mn[i,1] >= lat.Lm
            Neighbours_mn[i,1] -= lat.Lm
        end
    end
    # set the indices to start at one instead of zero for their use in julia indexing.
    Neighbours_mn[:,1:2] .+=1
    return Neighbours_mn
end

"""
This function returns a array containing the positions of the sites.
"""
function position_matrix(lat::Lattice)
    positionMatrix = zeros(Float64, (lat.dim_sub*2,2))
    for Pab=[0,1], m=1:lat.Lm, n=1:lat.Ln
        positionMatrix[index(m,n,Pab,lat;start=1),:]= particle_position(m,n,Pab,lat;start=1)
    end 
    return positionMatrix
end 


