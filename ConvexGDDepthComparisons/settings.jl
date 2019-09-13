include("../utils.jl")
using Random
using Distributed

Random.seed!(42)
#target which defines shape of the loss function
target = [.5 .1]

# initialization values to try
numinit = 20
initializations = randn(numinit, 2)
#number of seeds per model and initialization
numseed = 100

#depths to try (number of matrix multiplies)
depths = collect(1:25)
numdepth = length(depths)
#width of intermediate layers
width = 10

#training settings
lrvals = 10. .^ [-5, -4, -3, -2, -1]
numlr = length(lrvals)
tol = 1e-4
numstep=Int(1e9)

function filename(depthIndex::Int, initIndex::Int, lrindex::Int, seed::Int)
    "data/trajectory_depthindex_$(depthIndex)_initindex_$(initIndex)_lrindex_$(lrindex)_seed_$(seed).txt"
end

function savetrace(depthIndex::Int, initIndex::Int, lrindex::Int, seed::Int)
    #sizes of the LNN layers
    depth = depths[depthIndex]
    sizes = vcat([2], repeat([width], depth-1), [1])
    fname = filename(depthIndex, initIndex, lrindex, seed)
    lr = lrvals[lrindex]
    init = reshape(initializations[initIndex, :], (1, 2))
    saveconvexGDtrace(sizes, target, init, fname, 
                        lr=lr, tol=tol, numstep=numstep)
end

savetrace(indices::Tuple{Int, Int, Int, Int}) = savetrace(indices...)

