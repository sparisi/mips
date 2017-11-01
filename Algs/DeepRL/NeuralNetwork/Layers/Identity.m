classdef Identity < Layer
    
    methods

        function obj = Identity()
            obj.W = [];
        end
        
        function Y = forward(obj, X)
            Y = X;
        end
        
        function [dX, dW] = backward(obj, dY)
            dW = [];
            dX = dY;
        end
        
    end
    
end