%%% Gaussian Mixture Model with constant mean and covariance.
%%% Params: mean and covariance.
classdef gmm_constant
    
    properties(GetAccess = 'public', SetAccess = 'private')
        mu;    % means
        dim;   % length of the action drawn
        Sigma; % covariances
        p;     % mixing proportions
        gmax;  % max number of Gaussians
    end
    
    methods
        
        function obj = gmm_constant(mu, Sigma, p, gmax)
            obj.mu = mu;
            obj.Sigma = Sigma;
            obj.p = p;
            obj.gmax = gmax;
            obj.dim = size(mu,2);
        end
        
        function action = drawAction(obj)
            gmm = gmdistribution(obj.mu, obj.Sigma, obj.p);
            action = gmm.random()';
        end
        
        function probability = evaluate(obj, action)
            gmm = gmdistribution(obj.mu, obj.Sigma, obj.p);
            probability = gmm.pdf(action');
        end
        
        function S = entropy(obj)
            S = NaN;
        end
        
        function obj = makeDeterministic(obj)
            obj.Sigma = ones(size(obj.Sigma)) * 1e-8;
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            weights = max(weights,1e-8);
            weights = weights / sum(weights);
            [~, gmm] = emgm(Action, weights, obj.gmax);
            obj.p = gmm.ComponentProportion';
            obj.mu = gmm.mu';
            obj.Sigma = gmm.Sigma;
        end
        
        function obj = randomize(obj, factor)
            obj.Sigma = obj.Sigma .* factor;
        end
       
    end
    
end
