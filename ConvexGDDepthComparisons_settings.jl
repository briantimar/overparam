include("utils.jl")
using Random
using Distributed

Random.seed!(42)
#target which defines shape of the loss function
target = [.5 .1]

# initialization values to try
numinit = 5
initializations = randn(numinit, 2)
#number of seeds per model and initialization
numseed = 50

#depths to try (number of matrix multiplies)
depths = collect(1:50)
ndepth = length(depths)
#width of intermediate layers
width = 10

#training settings
lr = 1e-2
tol = 1e-4
numstep=50000

function filename(depthIndex::Int, initIndex::Int, seed::Int)
    "trajectory_depthindex_$(depthIndex)_initindex_$(initIndex)_seed_$(seed).txt"
end

function savetrace(depthIndex::Int, initIndex::Int, seed::Int)
    #sizes of the LNN layers
    depth = depths[depthIndex]
    sizes = vcat([2], repeat([width], depth-1), [1])
    fname = filename(depthIndex, initIndex, seed)
    init = reshape(initializations[initIndex, :], (1, 2))
    saveconvexGDtrace(sizes, target, init, fname, 
                        lr=lr, tol=tol, numstep=numstep)
end

savetrace(indices::Tuple{Int, Int, Int}) = savetrace(indices...)

