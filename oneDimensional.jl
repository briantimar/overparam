using Plots
using Flux
pyplot()

include("models.jl")
include("utils.jl")

loss(w) = (1 - prod(w))^2
lossfn(l::LNN) = ( () -> loss(contract(l)))

scale = 2
xvals = yvals = collect(-scale:.1:scale)
lims = (minimum(xvals), maximum(xvals))
lossgrid = gridloss(loss, xvals, yvals)

init = [-1., 1.]
initc = prod(init)
lnn1 = LNN([1, 1])
lnn2 = LNN([1, 1, 1])
trainto!(lnn1, reshape([initc], 1,1), tol=1e-4, numstep=10000)
trainto!(lnn2, reshape([initc], 1,1), tol=1e-4, numstep=10000)

lr = .01
optimizer = Descent(lr)
l1, c1 = gettrajectory!(lnn1, lossfn(lnn1), optimizer, 10000; tol=1e-4)
c1 = reshape(c1, (length(c1)))
x1 = rand(length(c1)) .+ .5
y1 = c1 ./ x1

x2 = Vector{Float64}()
y2 = Vector{Float64}()
function modelcallback(model)
    push!(x2, getdata(model.layers[1])[1])
    push!(y2, getdata(model.layers[2])[1])
end

l2, c2 = gettrajectory!(lnn2, lossfn(lnn2), optimizer, 10000; tol=1e-4, modelcallback=modelcallback)
c2 = reshape(c2, length(c2))

p = wireframe(xvals, yvals, lossgrid)
xmin = collect(.003:.003:scale)
ymin = 1 ./ xmin

toplot = abs.(ymin) .< scale
xmin = xmin[toplot]
ymin = ymin[toplot]
zmin = zeros(length(ymin))

plot!(p, xmin, ymin, zmin,
        color = :black,
        xlims = lims, ylims=lims, 
        linewidth=3)

plot!(p, -xmin, -ymin, zmin,
        color = :black,
        xlims = lims, ylims=lims, 
        linewidth = 3)

plot!(p, x1, y1, l1,
        marker=:circle,
        color=:blue)
plot!(p, x2, y2, l2,
        marker=:cross,
        color=:red)

## now on a one-dimensional surface
q = plot(xvals, map(loss, xvals), 
            color=:black)
plot!(q, c1, l1, 
        color=:blue, 
        marker=:circle, 
        width=0,
        label="1d")

plot!(q, c2, l2, 
        color=:red, 
        marker=:cross, 
        width=0,
        label="2d")

q2 = plot(diff(c1), 
            label="1d")
plot!(q2, diff(c2), 
            label="2d")