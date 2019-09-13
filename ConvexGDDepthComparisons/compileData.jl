#collect all the txt outputs into larger arrays, for convenience.
include("settings.jl")
using DelimitedFiles
using JLD

#to save space only pack a few traces per setting
numtrace=1

for ilr in 1:numlr
    @info "Collecting lr index $ilr..."
    alldata = Dict{String, Any}()
    for ii in 1:numinit
        for id in 1:numdepth
            traces = Vector{Array{Float64, 2}}()
            lengths = Vector{Int}()           
            for s in 1:numseed
                try
                    t = readdlm(filename(id, ii, ilr, s))
                    push!(lengths, length(t))
                    length(traces) < numtrace && push!(traces, t)
                catch
                    t = nothing
                    length(traces) < numtrace && push!(traces, t)
                end
                
            end
            alldata["traces_depthindex_$(id)_initindex_$(ii)"] = traces
            alldata["lengths_depthindex_$(id)_initindex_$(ii)"] = lengths
        end
    end
    for var in [:numseed, :numinit, :numdepth, :numlr, :initializations, :depths, :tol, :width, :numstep, :target]
        alldata["$var"] = eval(var)
    end
    alldata["lr"] = lrvals[ilr]
    save("data/alldata_lrindex_$(ilr).jld", alldata)
end

@info "Done"