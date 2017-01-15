classdef Tanh < Layer
    
    methods

        function obj = Tanh()
            obj.W = [];
        end
        
        function Y = forward(obj, X)
            Y = tanh(X);
        end
        
        function [dX, dW] = backward(obj, dY)
            dW = [];
            dX = (1 - obj.Y) .* dY;
        end
        
    end
    
end