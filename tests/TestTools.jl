

using LinearAlgebra, Test, Tables
include(abspath(@__DIR__,"../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))

function TestPermutation(par::Parameters, lat::Lattice)
    P = zeros((lat.Nt, lat.Nt))
    for t = 1:lat.Nt
        for tp = 1:lat.Nt
            t_min= t-1
            if tp==t_min #δ_{t',t-1}
                P[t,tp] = 1                
            elseif t_min==0 && tp==lat.Nt
                P[t,tp] = -1
            end 
        end 
    end 

    P_now = time_permutation_Matrix_anti_pbc(lat)
    @test P==P_now
end 

function TestTrace(par::Parameters, lat::Lattice)
    M = rand((lat.D,lat.D))
    @time trM = tr(M)
    @time trM_noisy = Trace_invD(M, lat.D, K =70)
    @test isapprox(trM,trM_noisy, atol=0.1)
end 

function TestStorage(par::Parameters, lat::Lattice)
    A = rand(ComplexF64,(lat.D, lat.D))
    Filename = "test"
    @time StoreResult(Filename, A)
    @time B = ReadResult(Filename, complex=true)
    @test A==B

    A = rand(lat.D, lat.D)
    Filename = "test"
    @time StoreResult(Filename, A)
    @time B = ReadResult(Filename, complex=false)
    @test A==B
end 

function TestFigureFunction(par::Parameters, lat::Lattice)
    ϕ = rand(lat.D)
    error = ones(lat.D)
    step = 1:1:lat.D
    CreateFigure(ϕ,"plots","testFigure",fmt="-", x_label="x", y_label=L"$\Delta$ niks")
    CreateFigure(ϕ,"plots","testFigure",fmt="*", y_err=error, x_label="x", y_label=L"$\Delta$ niks")
    CreateFigure(step, ϕ,"plots","testFigure", fmt="o", x_label="x", y_label=L"$\Delta$ niks")
    CreateFigure(step, ϕ,"plots","testFigure", y_err=error, fmt="o", x_label="x", y_label=L"$\Delta$ niks")
end

lat = Lattice(2,2,6)
par = Parameters(2.0, 0.0, 1.0, 0.5)
TestPermutation(par, lat)
TestStorage(par, lat)
TestFigureFunction(par, lat)
# TestTrace(par ,lat)