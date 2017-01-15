classdef Bias < Layer
    
    methods

        function obj = Bias(dimX)
            obj.W = mymvnrnd(0,1/dimX,dimX);
       end
        
        function Y = forward(obj, X)
            Y = bsxfun(@plus, X, obj.W);
        end
        
        function [dX, dW] = backward(obj, dY)
            dW = sum(dY, 1);
            dX = dY;
        end
        
    end
    
end
