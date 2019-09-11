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
getdata(l::Layer) = l.matrix.data

function LNN(sizes::Vector{Int})
    Chain((Layer(sizes[i], sizes[i+1]) for i in 1:(length(sizes)-1))...)
end

"Contracts an LNN to produce a single (dout, din) matrix"
function contract(model::Chain)
    W = getdata(model.layers[1])
    for i in 2:length(model)
        W = getdata(model.layers[i]) * W
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
    grads = Tracker.gradient(l2lossfn, params)
    for p in params
        Tracker.update!(optimizer, p, grads[p])
    end
end

