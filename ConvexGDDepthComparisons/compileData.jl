#collect all the txt outputs into larger arrays, for convenience.
include("settings.jl")
using DelimitedFiles
using JLD

alldata = Dict{String, Any}()

for ii in 1:numinit
    for id in 1:numdepth
        for ilr in 1:numlr
            traces = Vector{Union{Array{Float64, 2}, Nothing}}()
            for s in 1:numseed
                try
                    t = readdlm(filename(id, ii, ilr, s))
                catch
                    t = nothing
                end
                push!(traces, t)
            end
            alldata["traces_depthindex_$(id)_initindex_$(ii)_lrindex_$(ilr)"] = traces
        end
    end
end

for var in [:numseed, :numinit, :numdepth, :numlr :initializations, :depths, :lrvals, :tol, :width, :numstep, :target]
    alldata["$var"] = eval(var)
end

save("data/alldata.jld", alldata)