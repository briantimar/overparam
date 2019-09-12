#collect all the txt outputs into larger arrays, for convenience.
include("settings.jl")
using DelimitedFiles
using JLD

alldata = Dict{String, Any}()

for ii in 1:numinit
    for id in 1:ndepth
        traces = Vector{Array{Float64, 2}}()
        for s in 1:numseed
            t = readdlm(filename(id, ii, s))
            push!(traces, t)
        end
        alldata["traces_depthindex_$(id)_initindex_$(ii)"] = traces
    end
end

for var in [:numseed, :numinit, :ndepth, :initializations, :depths, :lr, :tol, :width, :numstep]
    alldata["$var"] = eval(var)
end

save("data/alldata.jld", alldata)