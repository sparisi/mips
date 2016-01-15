classdef GmmConstant < Policy
% GMMCONSTANT Gaussian Mixture Model with constant means and covariances.
% Parameters: means and covariances.
%
% As the number of components is variable (up to GMAX), this class does not
% implement the properties THETA and DPARAMS.
% Means are stored in rows, as in Matlab's GMDISTRIBUTION.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        mu    % means
        Sigma % covariances
        p     % mixing proportions
        gmax  % max number of Gaussians
    end
    
    methods

        %% Construction
        function obj = GmmConstant(varargin)
            if nargin == 3 % initialize with a single Gaussian
                mu0 = varargin{1};
                sigma0 = varargin{2};
                obj.gmax = varargin{3};
                
                n_params = length(mu0);
                obj.mu = zeros(obj.gmax,n_params);
                obj.Sigma = zeros(n_params,n_params,obj.gmax);
                for i = 1 : obj.gmax
                    obj.mu(i,:) = mymvnrnd(mu0,sigma0);
                    obj.Sigma(:,:,i) = sigma0;
                end
                obj.p = ones(obj.gmax,1) / obj.gmax;
            elseif nargin == 4 % directly init with all components
                obj.mu = varargin{1};
                obj.Sigma = varargin{2};
                obj.gmax = varargin{3};
                obj.p = varargin{4};
            else
                error('Wrong number of input arguments.')
            end
            [k,d] = size(obj.mu);
            [d1,d2,d3] = size(obj.Sigma);
            assert(d1 == d & d2 == d & d3 == k, ...
                'Dimensions are not consistent.');
            obj.daction = size(obj.mu,2);
        end

        %% Distribution functions
        function action = drawAction(obj,n)
            gmm = gmdistribution(obj.mu, obj.Sigma, obj.p);
            action = gmm.random(n)';
        end
        
        function probability = evaluate(obj, action)
            gmm = gmdistribution(obj.mu, obj.Sigma, obj.p);
            probability = gmm.pdf(action');
        end

        %% Entropy
        function S = entropy(obj)
            S = NaN;
        end

        %% WLM
        function obj = weightedMLUpdate(obj, weights, Actions)
            assert(min(weights >= 0), 'Weights cannot be negative.');
            [~, gmm] = emgm(Actions, obj.gmax, weights);
            obj.p = gmm.ComponentProportion';
            obj.mu = gmm.mu';
            obj.Sigma = gmm.Sigma;
        end

        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.Sigma = obj.Sigma * 1e-8;
        end
        
        function obj = randomize(obj, factor)
            obj.Sigma = obj.Sigma .* factor;
        end
       
    end
    
end
