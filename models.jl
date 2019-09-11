using LinearAlgebra
using Flux

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

function LNN(sizes::Vector{Int})
    Chain((Layer(sizes[i], sizes[i+1]) for i in 1:(length(sizes)-1))...)
end


