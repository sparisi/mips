classdef Lin < Layer
    
    methods

        function obj = Lin(dimX, dimY)
            obj.W = mymvnrnd(0,1/dimX,dimX*dimY);
        end
        
        function Y = forward(obj, X)
            dimX = size(X,2);
            dimY = numel(obj.W) / dimX;
            W = reshape(obj.W, dimX, dimY);
            Y = X * W;
        end
        
        function [dX, dW] = backward(obj, dY)
            dimY = size(dY,2);
            dimX = numel(obj.W) / dimY;
            W = (reshape(obj.W, dimX, dimY));
            dX = dY * W';
            dW = obj.X' * dY;
            dW = dW(:)';
        end
        
    end
    
end