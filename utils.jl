using Random
include("models.jl")

"Grid evaluation of 2D loss function."
function gridloss(lossfn, xvals::Vector{Float64}, yvals::Vector{Float64})
    nx = length(xvals)
    ny = length(yvals)
    outputs = zeros(Float64, nx, ny)
    for i = 1:nx, j=1:ny
        w = [xvals[i] yvals[j]]
        outputs[i,j] = lossfn(w)
    end
    return outputs
end

targetloss(w0::Union{Vector{Float64}, Array{Float64, 2}}) = ( w -> sum((w .- w0).^2))


function l2lossfn(x::Vector{Float64}, y::Vector{Float64})
    return w -> sum((y .- w * x).^2)
end

function getcoords(wvals::Vector{Array{Float64, N}}) where N
    length(wvals[1]) == 2 || error("Expecting 2d inputs")
    wx = zeros(Float64, length(wvals))
    wy = zeros(Float64, length(wvals))
    for i in 1:length(wvals)
        wx[i] = wvals[i][1]
        wy[i] = wvals[i][2]
    end
    return wx, wy
end

"Builds LNN which contracts to the given init array. Then performs GD to contract to the target."
function getconvexGDtrace(sizes::Vector{Int}, target::Array{Float64, 2}, init::Array{Float64, 2}; 
                           lr=1e-2, numstep=10000, tol=1e-3, seed=nothing)
    #build the model
    isnothing(seed) || Random.seed!(seed)
    model = LNN(sizes)
    #initialize 
    converged = trainto!(model, init, numstep=numstep, tol=tol)
    converged || error("Initialization failed.")
    # now train to the target.
    lossfn = targetloss(model, target)
    optimizer = Descent(lr)
    convcheck= sqrt
    losses, contractions = gettrajectory!(model, lossfn, optimizer, numstep, 
                                            tol=tol, convcheck=convcheck)
    return losses, contractions
end
