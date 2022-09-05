#=This file will contain the functions and struct for the square lattice.=#


"""
Lattice struct containing the lattice paramaters

Arguments:
    Lm (int)        The m length of the graphene lattice grid (horizontal length)
    Ln (int)        The n length of the graphene lattice grid (vertical length)
    Nt (int)        The number time steps 
    a  (Float)      The lattice spacing
    D  (int)        The total dimension of the field (Lm*Ln*Nt)
"""
struct Lattice
    Lm::Int
    Ln::Int
    Nt::Int
    a::Float64
    D::Int
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
function index(m, n, τ, lat::Lattice;start=0)
     #m,n and τ index starting at 0
    if start==0
        return m + n*lat.Lm + τ*(lat.Lm * lat.Ln) 
    elseif start==1
        index(m-1, n-1, τ-1, lat,start=0) +1
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
function index(m,n,lat;start=0)
    #m,n and index starting at 1 (equal time index (Nt=1))
    #if no Nt is given the index function for Nt=1 is used.
    if start==0
        return m + n*lat.Lm 
    elseif start==1
        return (m-1) + (n-1)*lat.Lm + 1 
    end 
end

