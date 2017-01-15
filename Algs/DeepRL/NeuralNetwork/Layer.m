classdef (Abstract) Layer < matlab.mixin.Heterogeneous & matlab.mixin.Copyable
% LAYER Neural network layer. It can store the last input at each forward 
% pass for faster backward pass updates.
    
    properties
        W % Parameters of the layer
        X % Last input of the layer
        Y % Last output of the layer
    end
    
    methods
        
        Y = forward(obj, X);
        % Activation function. It produces the output of the layer.
        
        function Y = forwardfull(obj, X)
        % In addition, this function stores input X and output Y.
            Y = obj.forward(X);
            obj.X = X;
            obj.Y = Y;
        end
        
        [dX, dW] = backward(obj, dY);
        % It computes the loss gradient wrt the input (received from the
        % last forward pass) and the weights, given the backpropagated
        % gradient from the next layer.
        
    end
    
end