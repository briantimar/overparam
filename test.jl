include("utils.jl")
include("models.jl")
using Plots


loss = targetloss([.4, .4])
xvals = [x for x in -2:.1:2]
yvals = copy(xvals)

lossvals = gridloss(loss, xvals, yvals)

model1 = LNN([2, 1])
optimizer = Descent(.01)
steps = 100
modelloss = () -> loss(contract(model1))
trlosses, contractions = gettrajectory!(model1, modelloss, optimizer, steps)
wx, wy = getcoords(contractions)

pyplot()
p = wireframe(xvals, yvals, lossvals);
plot!(p, wx,wy, trlosses,color=:black)