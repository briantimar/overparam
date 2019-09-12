include("utils.jl")
include("models.jl")
using Plots
using Random

#sampling of loss values to define a surface
target = [.5 .5]
loss = targetloss(target)
xvals = [x for x in -2:.1:2]
yvals = copy(xvals)
lossvals = gridloss(loss, xvals, yvals)

#now intialize LNNs at a fixed starting point ...
start = randn(1,2)
sizes = [[2, 1], [2, 4, 1], [2, 4, 4, 1], [2, 4, 4, 4, 1]]
ns, tol = 10000, 1e-3

losses = []
contractions = []
colors = [:red, :green, :blue, :purple]
lengths = []
p = wireframe(xvals, yvals, lossvals);

for i in 1:length(sizes)
    l, c = getconvexGDtrace(sizes[i], target, start, 
                            lr=1e-2, numstep=ns, tol=tol, seed=1)
    push!(losses, l)
    push!(contractions, c)
    push!(lengths, length(l))

    wx, wy = getcoords(c)
    plot!(p, wx, wy, l, marker=:circle, color=colors[i])
end

display(p)