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

targetloss(w0::Vector{Float64}) = ( w -> sum((w .- w0).^2))

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