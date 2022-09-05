#= 
this file contains the tests for the hexagonal lattice functions.
=#
include(abspath(@__DIR__,"../src/hexagonalLattice.jl"))
include(abspath(@__DIR__, "../src/tools.jl"))
using Graphs, GraphPlot, Compose, Colors, Cairo, Test
function Test_NeighbourMatrixVisual(par::Parameters, lat::Lattice)
    neighbours = neighbour_matrix(lat)
    @test isequal(neighbours,neighbours')
    plot_matrix(neighbours, "neighbour_matrix")

    g = Graph(neighbours)
    membership = ones(Int, lat.dim_sub*2)
    membership[1:lat.dim_sub].= floor(Int,2)
    nodecolor = [colorant"lightseagreen", colorant"orange"]
    # membership color
    nodefillc = nodecolor[membership]
    positionMatrix = position_matrix(lat)
    Compose.draw(PNG(abspath(@__DIR__,"../plots/connectionGraph.png")),gplot(g, positionMatrix[:,1], positionMatrix[:,2], nodefillc=nodefillc))
end


function Test_NeighbourMatrixTransitivity(par::Parameters, lat::Lattice)
    neighbours = neighbour_matrix(lat)
    connections = sum(neighbours,dims=1)
    @test minimum(connections)==maximum(connections)
end

function Test_NeighbourSpecific(par::Parameters, lat::Lattice)
    positionMatrix = position_matrix(lat)
    v_a = sqrt(3)*lat.a*[1,0]
    v_b = sqrt(3)*lat.a*[1,sqrt(3)]/2
    r_O = zeros((3,2))
    r_O[1,:] =-1/3*v_a  - 1/3*v_b
    r_O[2,:] = 2/3*v_a - 1/3*v_b
    r_O[3,:] = -1/3*v_a + 2/3*v_b
    clf()
    scatter(positionMatrix[1:lat.dim_sub,1],positionMatrix[1:lat.dim_sub,2],label="A")
    scatter(positionMatrix[lat.dim_sub+1:2*lat.dim_sub,1],positionMatrix[lat.dim_sub+1:2*lat.dim_sub,2],label="B")
    x= positionMatrix[index(3,3,0,lat,start=1),:]
    scatter(x[1],x[2],label="origin")
    scatter(r_O[:,1].+x[1],r_O[:,2].+x[2],label= "analytical neighbours")
    
    legend()
    axis("equal")
    xlabel("x")
    ylabel("y")
    savefig(abspath(@__DIR__,"../plots/lattice.png"))   


end 

par=Parameters(2.0,0.0,1.0,0.5)
lat = Lattice(20,6,6)

Test_NeighbourMatrixVisual(par,lat)
Test_NeighbourMatrixTransitivity(par,lat)
Test_NeighbourSpecific(par,lat)
