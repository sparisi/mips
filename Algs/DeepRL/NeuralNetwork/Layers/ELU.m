classdef ELU < Layer
    
    properties
        alpha
    end
    
    methods
        
        function obj = ELU(alpha)
            obj.W = [];
            if nargin == 0, alpha = 1; end
            obj.alpha = alpha;
        end
        
        function Y = forward(obj, X)
            Y = X;
            idx = X < 0;
            Y(idx) = obj.alpha * exp(X(idx) - 1);
        end
        
        function [dX, dW] = backward(obj, dY)
            dX = ones(size(obj.X));
            idx = obj.X < 0;
            dX(idx) = obj.alpha * exp(obj.X(idx));
            dX = dX .* dY;
            dW = [];
        end
        
    end
    
end