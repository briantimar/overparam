using LinearAlgebra: norm
using Flux
import Flux: params

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

function LNN(sizes::Vector{Int})
    Chain((Layer(sizes[i], sizes[i+1]) for i in 1:(length(sizes)-1))...)
end

"Contracts an LNN to produce a single (dout, din) matrix"
function contractdata(model::Chain)
    W = getdata(model.layers[1])
    for i in 2:length(model)
        W = getdata(model.layers[i]) * W
    end
    W
end 

function contract(model::Chain)
    W = model.layers[1].matrix
    for i in 2:length(model)
        W = model.layers[i].matrix * W
    end
    W
end

inputdim(l::Chain) = l.layers[1].din
outputdim(l::Chain) = l.layers[end].dout

"Loss function defined by a single input/output pair"
function l2lossfn(model, input::Vector{Float64}, output::Vector{Float64})
    length(input) != inputdim(model) && throw(ArgumentError("Invalid input dim"))
    length(output) != outputdim(model) && throw(ArgumentError("Invalid output dim"))
    return () -> sum((model(input) .- output).^2)
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
contraction."
function gettrajectory!(model::Chain, lossfn,
                        optimizer, numstep::Int)
    
    lossvals = Vector{Float64}()
    contractions = Vector{Array{Float64, 2}}()

    parameters = params(model)
    
    push!(lossvals, getdata(lossfn()))
    push!(contractions, contractdata(model))
    for step in 1:numstep
        dostep!(lossfn, parameters, optimizer)
        push!(lossvals, getdata(lossfn()))
        push!(contractions, contractdata(model))
    end
    return lossvals, contractions
end

