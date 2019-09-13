using Distributed
@everywhere include("settings.jl")
#collect all training settings
indices = [(di, ii, il, s) for di in 1:numdepth for ii in 1:numinit for il in 1:numlr for s in 1:numseed]
pmap(savetrace, indices)
