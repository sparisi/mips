classdef Sig < Layer
    
    methods

        function obj = Sig()
            obj.W = [];
        end
        
        function Y = forward(obj, X)
            Y = 1 ./ (1 + exp(-X));
        end
        
        function [dX, dW] = backward(obj, dY)
            dW = [];
            dX = dY .* obj.Y .* (1 - obj.Y);
        end
        
    end
    
end