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
mutable struct Lattice
    Lm::Int
    Ln::Int
    Nt::Int
    a::Float64
    dim_sub::Int
    D::Int
end 

# automatically compute the lattice dimension when the struct is created.
# set a to 0.71*10^(-3) eV-1 which is equvilant to 1.42 Å
Lattice(Lm,Ln,Nt) = Lattice(Lm,Ln,Nt,0.00071)
# set dimension of the sublatice to Lm*Ln
Lattice(Lm,Ln,Nt,a) = Lattice(Lm,Ln,Nt,a,Lm*Ln)
# set the total dimension D = 2*dim_sub*Nt
Lattice(Lm,Ln,Nt,a,dim_sub) = Lattice(Lm,Ln,Nt,a,dim_sub,dim_sub*2*Nt)

"""
Function to change the lattice struct mutable struct.

Input:
    lat (Lattice struct)      The lattice struct contianing the lattice parameters
    Lm (int)                  The m length of the graphene lattice grid (horizontal length)
    Ln (int)                  The n length of the graphene lattice grid (vertical length)
    Nt (int)                  The number time steps 
"""
function change_lat(lat::Lattice; Nt= 0, Lm=0, Ln=0)
    if Nt!==0
        lat.Nt = Nt
    end 
    if Lm!==0
        lat.Lm = Lm
    end 
    if Ln!==0
        lat.Lm=Lm
    end 
    lat.D = lat.dim_sub*lat.Nt*2
    lat.dim_sub = lat.Lm*lat.Ln
end 

"""
Particle struct indicating the index coordinates of a particle.

Input: 
    m (Int)     The Lm direction index on the sublattice
    n (Int)     The Ln direction index on the sublattice
    Pab (int (either 0 or 1)) The boolean index indicating whether the particle is on sublattice A or B (indicated by 0 or 1 respectively)
    start       The indication on how the indices are store (starting with either 0 or 1 as the first index)
"""
struct Particle
    m::Int
    n::Int
    Pab::Int
    start::Int
end 

#set the default start to 0
Particle(m, n, Pab) = Particle(m, n, Pab, 0) 

"""
Function to determine the particle position in real space.

Input: 
    particle (Particle struct)      The particle struct indicating which particles position to determine (struct contains m,n,Pab and start)
    lat (Lattice struct)            The lattice struct contianing the lattice parameters
Output:
    coordinate (2D Float)           The coordinates of the given particle
"""
function particle_position(particle::Particle, lat::Lattice)
    return particle_position(particle.m, particle.n, particle.Pab, lat, start=particle.start)
end 

"""
Function to determine the particle position in real space.

Input: 
    m (Int)     The Lm direction index on the sublattice
    n (Int)     The Ln direction index on the sublattice
    Pab (int (either 0 or 1)) The boolean index indicating whether the particle is on sublattice A or B (indicated by 0 or 1 respectively)
    lat (Lattice struct)            The lattice struct contianing the lattice parameters
Optional: 
    start       The indication on how the indices are store (starting with either 0 or 1 as the first index)(default=0)
Output:
    coordinate (2D Float)           The coordinates of the given particle
"""
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

"""
Function to determine the coordinates of the neighbour of particle at position x on sublattice Pab.

Input:
    x (2 Floats)    The coordinates of the particle
    Pab (Bool)              The boolean indicator on which lattice we are if Pab=0 then sublatice A elseif Pab=1 on sublatice B
    lat (Lattice struct)    The lattice struct contianing the lattice parameters
Output:
    Neighbours (3,2 floats) The real space coordinates of the three graphene neighbours.
"""
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


"""
Function to obtain the distance matrix of the particles. This matrix is used for the potential components

Input: 
    lat (Lattice struct)        Struct containing the lattice paramaters

Output:
    r   (Lat.dim_sub*2,Lat.dim_sub*2 matrix)    Float matrix returning the distance between all particles with eachother.
"""
function distance_matrix(lat::Lattice)
    x = position_matrix(lat)
    #create interaction matrix
    dim = lat.dim_sub * 2
    #ensure periodic boundary conditions are met
    Xmax = findmax(x[:,1])[1]+sqrt(3)/2*lat.a
    Ymax = findmax(x[:,2])[1]+lat.a/2
    r = distanceXY(x, dim, Xmax, Ymax)
    return r
end 