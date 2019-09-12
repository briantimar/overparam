using LinearAlgebra: norm
using Flux
import Flux: params
import Base: length

struct Layer
    din::Int
    dout::Int
    matrix::TrackedArray
    function Layer(din::Int, dout::Int)
        matrix = param(randn(dout, din) / sqrt(dout))
        new(din, dout, matrix)
    end
end

(l::Layer)(x) = l.matrix * x
Flux.@treelike Layer

getdata(l::Layer) = copy(l.matrix.data)
getdata(x::Tracker.TrackedReal) = copy(x.data)
getdata(x) = x


struct LNN
    sizes::Vector{Int}
    inputdim::Int
    outputdim::Int
    chain
    layers
    function LNN(sizes::Vector{Int})
        chain=Chain((Layer(sizes[i], sizes[i+1]) for i in 1:(length(sizes)-1))...)
        inputdim = sizes[1]
        outputdim = sizes[end]
        layers=chain.layers
        new(sizes, inputdim, outputdim, chain, layers)
    end
end

(l::LNN)(x) = l.chain(x)
getindex(l::LNN, i::Int) = l.chain.layers[i]
inputdim(l::LNN) = l.inputdim
outputdim(l::LNN) = l.outputdim
length(l::LNN) = length(l.chain)
Flux.@treelike LNN

"Contracts an LNN to produce a single (dout, din) matrix"
function contractdata(model::LNN)
    W = getdata(model.layers[1])
    for i in 2:length(model)
        W = getdata(model.layers[i]) * W
    end
    W
end 

function contract(model::LNN)
    W = model.layers[1].matrix
    for i in 2:length(model)
        W = model.layers[i].matrix * W
    end
    W
end

"Loss function defined by a single input/output pair"
function l2lossfn(model::LNN, input::Vector{Float64}, output::Vector{Float64})
    length(input) != inputdim(model) && throw(ArgumentError("Invalid input dim"))
    length(output) != outputdim(model) && throw(ArgumentError("Invalid output dim"))
    return () -> sum((model(input) .- output).^2)
end

"Loss based on norm to target weights"
function targetloss(model::LNN, w0::Union{Vector{Float64}, Array{Float64, 2}}) 
    () -> sum((w0 .- contract(model)).^2)
end

"Compute gradients of lossfn WRT parameters given, then update them with optimizer 
(this latter step zeros the grads)"

function dostep!(lossfn, params, optimizer)
    grads = Tracker.gradient(lossfn, params)
    for p in params
        Tracker.update!(optimizer, p, grads[p])
    end
end

"Trains a linear neural net via gradient descent. At each training step, logs the loss function and 
contraction. "
function gettrajectory!(model::LNN, lossfn,
                        optimizer, numstep::Int; tol=-1.0, convcheck=identity)
    
    lossvals = Vector{Float64}()
    contractions = Vector{Array{Float64, 2}}()

    parameters = params(model)
    
    push!(lossvals, getdata(lossfn()))
    push!(contractions, contractdata(model))
    for step in 1:numstep
        dostep!(lossfn, parameters, optimizer)
        push!(lossvals, getdata(lossfn()))
        push!(contractions, contractdata(model))
        if convcheck(lossvals[end]) < tol
            @info "Tolerance $tol reached, halting training."
            return lossvals, contractions
        end
    end
    return lossvals, contractions
end

"Train the LNN to contract to the target array"
function trainto!(l::LNN, target::Array{Float64, 2}, optimizer; numstep::Int=1000, tol=1e-3)
    size(target) == (outputdim(l), inputdim(l)) || error("LNN and target dimensions do not match.")
    lossfn = () -> sum( (target .- contract(l)).^2)
    convcheck = (x -> sqrt(abs(x)))
    return gettrajectory!(l, lossfn, optimizer, numstep, tol=tol, convcheck=convcheck)
end

function trainto!(l::LNN, target::Array{Float64, 2}; lr=1e-3, tol=1e-3, numstep::Int=1000)
    optimizer = ADAM(lr)
    lossvals, contractions = trainto!(l, target, optimizer, numstep=numstep, tol=tol)
    converged = abs(lossvals[end]) < tol
    converged || @warn "Failed to converge."
    return converged
end

