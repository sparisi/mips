classdef ReLU < Layer
    
    methods
        
        function obj = ReLU()
            obj.W = [];
        end
        
        function Y = forward(obj, X)
            Y = bsxfun(@max,0,X);
        end
        
        function [dX, dW] = backward(obj, dY)
            dX = bsxfun(@gt,obj.X,0) .* dY;
            dW = [];
        end
        
    end
    
end