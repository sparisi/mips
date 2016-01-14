classdef (Abstract) Gaussian < Policy
% GAUSSIAN Generic class for Gaussian distributions.
    
    methods
        
        function S = entropy(obj, varargin)
        % Differential entropy, can be negative
            S = 0.5*log( (2*pi*exp(1))^obj.daction * det(obj.Sigma) );
        end

    end

end
