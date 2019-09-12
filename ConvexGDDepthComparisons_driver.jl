using Distributed
@everywhere include("ConvexGDDepthComparisons_settings.jl")
#collect all training settings
indices = [(di, ii, s) for di in 1:ndepth for ii in 1:numinit for s in 1:numseed]
pmap(savetrace, indices[1:2])
