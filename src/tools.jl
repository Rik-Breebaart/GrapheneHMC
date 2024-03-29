#=This script will contain al the general tools needed for the simulations=#
using LinearAlgebra, PyPlot, Random, IterativeSolvers, DelimitedFiles, DataFrames, CSV
include(abspath(@__DIR__, "../src/hybridMonteCarlo.jl"))

mutable struct Parameters
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

function change_mass(mass, par::Parameters)
    par.mass = mass
end 

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
    AB = kron(I, array[indA, indB])
    BB = kron(I, array[indB, indB])
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
    return R
end 


"""
Function to determine the trace of the inverse of matrix D using the noisy estimator method.

Input:
    D   (Matrix)    The matrix of which the inverse trace is computed
    dim (Int)       The size of the matrix
    K   (Int)       The number of random vectors are used to compute the noisy estimator
Optional:
    rng             The random number generator used (Default = MersenneTwister() )

Output:
    Trace(inv(D))   The noisy estimator of the trace of inv(D)
"""
function Trace_invD(D, dim; K=10, rng=MersenneTwister())
    ξ = randn(rng,ComplexF64, (K,dim))
    trace_invD=0
    invD = inv(D)
    for k = 1:K
        trace_invD += adjoint(ξ[k,:])*invD*ξ[k,:]
    end
    return trace_invD/K
end 


"""
Function to store results to a file.

Input:
    Filename (string)   The filename and folder to which the result is stored (starting within the results folder)
    result (array)      The array which is stored (can be either 1 or 2 dimensional)    
"""
function StoreResult(Filename, result; append=false)
    if append==false
        Dim = length(size(result))
        if Dim>2 
            error("The storage method is unable to store higher dimensional arrays (3 or more dimensions)")
        end 
        writedlm(abspath(@__DIR__,string("../results/",Filename,".txt")), result) 
    else
        io = open(abspath(@__DIR__,string("../results/",Filename,".txt")), "a")
        writedlm(io,result)
        close(io)
    end        
end


"""
Function to read the stored results from a file.

Input:
    Filename (string)   The filename and folder to which the result is stored (starting within the results folder)
Optional:
    Complex (bool)      Indicates whether the result in the stored file are complex or not.
"""
function ReadResult(Filename; complex=false)
    if complex==false
        result = readdlm(abspath(@__DIR__,string("../results/",Filename,".txt")))
    else 
        result = readdlm(abspath(@__DIR__,string("../results/",Filename,".txt")),'\t',Complex{Float64})       
    end
    return result
end 

"""
Function to create a PyPlot figure (to decluter application scripts and store to results folder)

Input:
    x    (Array)        The x coordinates for the plot
    y    (array)        The y coordinates for the plot
    Folder  (string)    THe folder inside the "results" folder to which the plot should be stored
    Filename   (string) The filename of the figure
Optional:
    y_err   (array)     The error of the results on the y-coordinates (default: nothing)
    fmt     (string)    The marker method used for the plot (default: ".")
    x_label (string)    The x axis label (default: nothing)
    y_label (string)    The y axis label (default: nothing)
    figure_title (string) The figure title (default: nothing)
"""
function CreateFigure(x, y, Folder, Filename,;y_err=nothing, fmt=".", x_label=nothing, y_label=nothing, figure_title=nothing)
    clf()
    if y_err===nothing
        plot(x,y,fmt)
    else 
        errorbar(x,y,yerr=y_err,fmt=fmt)
    end
    if x_label!==nothing
        xlabel(x_label)
    end 
    if y_label!==nothing
        ylabel(y_label)
    end
    if figure_title!==nothing
        title(figure_title)
    end
    savefig(abspath(@__DIR__,string("../results/",Folder,"/",Filename,".png")))
end

"""
Function to create a PyPlot figure (to decluter application scripts and store to results folder)
if no x array is provided.

Input:
    y    (array)        The y coordinates for the plot
    Folder  (string)    THe folder inside the "results" folder to which the plot should be stored
    Filename   (string) The filename of the figure
Optional:
    y_err   (array)     The error of the results on the y-coordinates (default: nothing)
    fmt     (string)    The marker method used for the plot (default: ".")
    x_label (string)    The x axis label (default: nothing)
    y_label (string)    The y axis label (default: nothing)
    figure_title (string) The figure title (default: nothing)
"""
function CreateFigure(y, Folder, Filename,;y_err=nothing, fmt=".", x_label=nothing, y_label=nothing, figure_title=nothing)
    clf()
    if y_err===nothing
        plot(y,fmt)
    else 
        errorbar(1:length(y)[1],y,yerr=y_err,fmt=fmt)
    end
    if x_label!==nothing
        xlabel(x_label)
    end 
    if y_label!==nothing
        ylabel(y_label)
    end
    if figure_title!==nothing
        title(figure_title)
    end
    savefig(abspath(@__DIR__,string("../results/",Folder,"/",Filename,".png")))
end        



function storage_folder(observable, equation, par::Parameters, lat::Lattice)
    folder = string(observable,"_eq",equation,"_beta_",floor(Integer,par.β*10),"Lm_",lat.Lm,"Ln_",lat.Ln,"Nt_",lat.Nt)
    S_folder_set(runNumber) = abspath(@__DIR__,string("../results/",folder,"_",runNumber))
    #create new folder if folder already exists
    runNumber = 0
    if isdir(S_folder_set(runNumber))
        while isdir(S_folder_set(runNumber))
            runNumber +=1 
        end        
    end
    mkdir(S_folder_set(runNumber))
    return string(folder,"_",runNumber)
end 

function storage_folder(observable, equation, β, lat::Lattice)
    folder = string(observable,"_eq",equation,"_beta_",floor(Integer,β*10),"Lm_",lat.Lm,"Ln_",lat.Ln,"Nt_",lat.Nt)
    S_folder_set(runNumber) = abspath(@__DIR__,string("../results/",folder,"_",runNumber))
    #create new folder if folder already exists
    runNumber = 0
    if isdir(S_folder_set(runNumber))
        while isdir(S_folder_set(runNumber))
            runNumber +=1 
        end        
    end
    mkdir(S_folder_set(runNumber))
    return string(folder,"_",runNumber)
end 


"""
Function to initialize a file and store the run paramaters. 
The function creates a file according to filename containing the paramaters of the simulation. 
This file can then be used to append the specific results to.

Input: 
    Filename (string + storage type (e.g. ".txt"))  The Filename with corresponding file type
    lat     (Lattice struct)                        The lattice struct containing the lattice parameters
    par     (Paramater struct)                      The paramater struct containing the fixed paramaters of the simulation (physical constants etc.)
    method  (Optional, string:"w","a") Default: "w" Optional method string specifing the method for open (see https://docs.julialang.org/en/v1/base/io-network/ for specifc methods)

"""
function Store_Settings(File_path, HMC_par::HMC_Parameters; method="w")
    # store configuration settings in a file inside the desired folder
    df = DataFrame(variable = ["Nsamples", "path_length", "step_size", "offset", "m_sw", "burn_in", "equation"], 
                   value= [HMC_par.Nsamples, HMC_par.path_length, HMC_par.step_size, HMC_par.offset, HMC_par.m_sw, HMC_par.burn_in, HMC_par.equation]
                   )
    if method=== "a"
        append=true
    else
        append=false
    end
    CSV.write(File_path, df, append=append)
end 

function Store_Settings(File_path, lat::Lattice; method="w")
    # store configuration settings in a file inside the desired folder
    df = DataFrame(variable = ["Lm", "Ln", "Nt", "a"], 
                   value= [lat.Lm, lat.Lm, lat.Nt, lat.a]
                   )
    if method=== "a"
        append=true
    else
        append=false
    end
    CSV.write(File_path, df, append=append)
end 

function Store_Settings(File_path, par::Parameters; method="w")
    # store configuration settings in a file inside the desired folder 
    df = DataFrame(variable = ["beta", "mass", "epsilon", "R0", "e2" , "kappa"], 
                value= [par.β, par.mass, par.ϵ, par.R0, par.e2, par.κ])
    if method=== "a"
        append=true
    else
        append=false
    end
    CSV.write(File_path, df, append=append)
end 

function par_from_df(df)
    if "beta" in df.variable
        par = Parameters(df[df.variable.=="beta",:].value[1], 
                        df[df.variable.=="mass",:].value[1], 
                        df[df.variable.=="epsilon",:].value[1], 
                        df[df.variable.=="R0",:].value[1],
                        df[df.variable.=="e2",:].value[1], 
                        df[df.variable.=="kappa",:].value[1])
        return par
    else 
        return nothing
    end
end

function lat_from_df(df)
    if "Lm" in df.variable
        lat = Lattice(df[df.variable.=="Lm",:].value[1], 
                        df[df.variable.=="Ln",:].value[1], 
                        df[df.variable.=="Nt",:].value[1], 
                        round(df[df.variable.=="a",:].value[1],digits=10))
        return lat
    else 
        return nothing
    end
end

function HMC_from_df(df)
    if "Nsamples" in df.variable
        HMC_par = HMC_Parameters(df[df.variable.=="Nsamples",:].value[1], 
                        df[df.variable.=="path_length",:].value[1], 
                        df[df.variable.=="step_size",:].value[1], 
                        df[df.variable.=="offset",:].value[1],
                        df[df.variable.=="m_sw",:].value[1], 
                        df[df.variable.=="burn_in",:].value[1],
                        df[df.variable.=="equation",:].value[1])
        return HMC_par
    else
        return nothing
    end
end

function Read_Settings(File_path)
    df = CSV.read(File_path, DataFrame)   
    return df 
end

function Read_Settings(File_path, structs)
    df = CSV.read(File_path, DataFrame)   
    a_lat = "lat" in structs
    a_hmc = "hmc" in structs
    a_par = "par" in structs
    if a_hmc && a_lat && a_par
        return lat_from_df(df), par_from_df(df), HMC_from_df(df)
    elseif a_hmc && a_lat
        return lat_from_df(df), HMC_from_df(df)
    elseif a_hmc && a_par
        return par_from_df(df), HMC_from_df(df)
    elseif a_par && a_lat
        return lat_from_df(df), par_from_df(df)
    elseif a_hmc 
        return HMC_from_df(df)
    elseif a_lat
        return lat_from_df(df)
    elseif a_par
        return par_from_df(df)
    else 
        error("Incorrect structs where provided should be either 'hmc', 'lat' or 'par' or a tuble of multiple of them.")
    end
end 
