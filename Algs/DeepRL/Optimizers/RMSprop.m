classdef RMSprop < handle
    
    properties
        alpha = 1e-3;
        gamma = 0.9;
        beta = 0.7;
        epsilon = 1e-8;
        m = 1;
        v = 0;
    end
    
    methods
        
        function obj = RMSprop(dim)
            obj.m = ones(1,dim);
            obj.v = zeros(1,dim);
        end
        
        function x = step(obj, x, dx)
            x = x - obj.beta * obj.v; % Nesterov momentum
            obj.m = obj.gamma * obj.m + (1 - obj.gamma) * dx.^2;
            obj.v = obj.beta * obj.v + obj.alpha ./ sqrt(obj.m + obj.epsilon) .* dx;
            x = x - obj.v;
        end
        
    end
    
end