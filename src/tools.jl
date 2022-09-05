#=This script will contain al the general tools needed for the simulations=#
using LinearAlgebra, PyPlot

struct Parameters
    β::Float64
    mass::Float64
    ϵ::Float64
    R0::Float64
    e2::Float64
    κ::Float64
end 

# the tight binding constant and gaussian system coulomb potential e^2 are fixed
#κ = 2.8 abd e^2 = 1/137
Parameters(β, mass, ϵ, R0) = Parameters(β, mass, ϵ, R0, 1.0/137, 2.8) 

"""
Function to create time permutation matrix which shifts field site index at t to t-1.
For anti-periodic boundary conditions in time.
δ_{t-1,t}

Input:
    lat (Lattice)   The lattice struct containing the lattice paramaters
Output
    P  (DxD array of booleans)  The permutation matrix 
"""
function time_permutation_Matrix_anti_pbc(lat::Lattice)
    I = Diagonal(ones(lat.Nt))
    P = Transpose(I[[2:lat.Nt; 1],:])
    # anti periodic boundary conditions 
    P[1,lat.Nt]=-1
    return P
end

"""
Function to create time permutation matrix which shifts field site index at t to t-1. 
For periodic boundary conditions in time

Input:
    lat (Lattice)   The lattice struct containing the lattice paramaters
Output
    P  (Nt x Nt array of booleans)  The permutation matrix 
"""
function time_permutation_Matrix_pbc(lat::Lattice)
    I = Diagonal(ones(lat.Nt))
    P = Transpose(I[[2:lat.Nt; 1],:])
    # this is the permutation with periodic boundary conditions
    return P
end

"""
Function to convert a Lm*Ln*2 matrix to a D dimensional matrix where the A and B sublattice
components are correctly possitioned according to indexing with each at equal time 

Input:
    lat (Lattice)   The lattice struct containing the lattice paramaters
    arrayAB         The position dependent array to be distributed with equal time (first Lm*ln for sublattice A, second Lm*Ln for B)

Output:
    B   (DxD array) Array with arrayAB distirbuted over equal time (such that the first Nt*Lm*Ln are sublattice A and second B)
"""
function Array_Equal_Time(array, lat::Lattice)
    indA = 1:lat.dim_sub
    indB = lat.dim_sub+1:2*lat.dim_sub
    I = Diagonal(ones(lat.Nt))
    AA = kron(I, array[indA, indA])
    BB = kron(I, array[indA, indB])
    AB = kron(I, array[indB, indB])
    BA = kron(I, array[indB, indA])
    return [AA AB; BA BB]
end

"""
Function to convert a Lm*Ln*2 matrix to a D dimensional matrix where the A and B sublattice
components are correctly possitioned according to indexing with permutated time δ_{t-1,1} using anti periodic boundary conditions 

Input:
    lat (Lattice)   The lattice struct containing the lattice paramaters
    arrayAB         The position dependent array to be distributed with equal time (first Lm*ln for sublattice A, second Lm*Ln for B)

Output:
    B   (DxD array) Array with arrayAB distirbuted over equal time (such that the first Nt*Lm*Ln are sublattice A and second B)
"""
function Array_Permute_Time_t1_t(array, lat::Lattice)
    indA = 1:lat.dim_sub
    indB = lat.dim_sub+1:2*lat.dim_sub
    P = time_permutation_Matrix_anti_pbc(lat)
    AA = kron(P, array[indA, indA])
    AB = kron(P, array[indA, indB])
    BA = kron(P, array[indB, indA])
    BB = kron(P, array[indB, indB])
    return [AA AB; BA BB]
end

"""
Function to show a matrix as a plot.

Input:
    A   (2D array)     Array to be displayed (should be 2D)
    filename    (string)    The name of the stored figure
    folder (string)     The folder to which the figures is stored

Output: 
    nothing             THe function stores a figure in the plots folder or another prescribed folder
"""
function plot_matrix(A, filename; title=nothing, folder=abspath(@__DIR__,"../plots"))
    clf()
    imshow(real.(A),cmap=ColorMap("gray"))
    if title!==nothing
        title(title)
    end 
    colorbar()
    savefig(abspath(folder,filename*".png"))
end 


"""
A function to create a meshgrid array of the distances between particles in a 2d space with periodic boundary conditions.

Input: 
    arrayXY (2xN array of Floats)     The array contianing the x and y coordinates of N particles
    N       (Int)                     The number of particles
    Xmax    (Float)                   The maximum x of the square box
    Ymax    (Float)                   The maximum y of the square box

Output:
    R  (NxN array of Floats)     Array of the  euclidean distance between the particles on the array given by sqrt(x^2+y^2)
"""
function distanceXY(arrayXY, N, Xmax, Ymax)
    #create distance matrices
    CoordX = [arrayXY[i,1]-arrayXY[j,1] for i in 1:N, j in 1:N]
    CoordY = [arrayXY[i,2]-arrayXY[j,2] for i in 1:N, j in 1:N]
    #ensure periodic boundary conditions are met
    CoordX[CoordX .> Xmax/2] .-=Xmax
    CoordX[CoordX .< -Xmax/2] .+=Xmax
    CoordY[CoordY .> Ymax/2] .-=Ymax
    CoordY[CoordY .< -Ymax/2] .+=Ymax
    # Create euclidean distance matrix
    R = sqrt.(CoordX[:,:].^2 + CoordY[:,:].^2)
end 
