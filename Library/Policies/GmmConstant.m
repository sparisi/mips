classdef GmmConstant < Policy
% GMMCONSTANT Gaussian Mixture Model with constant means and covariances.
% Parameters: means, covariances and mixture weights.
% Means are stored in rows, as in Matlab's GMDISTRIBUTION.
%
% As the number of components is variable (up to GMAX), this class does not
% implement the properties THETA and DPARAMS.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        mu    % means
        Sigma % covariances
        p     % mixing proportions
        gmax  % max number of Gaussians
    end
    
    methods

        %% Constructor
        function obj = GmmConstant(varargin)
            if nargin == 3 % Initialize with a single Gaussian
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
            elseif nargin == 4 % Directly init with all components
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

        %% GMM.RANDOM
        function action = drawAction(obj,n)
            components = mymnrnd(obj.p,n); % Select the Gaussians for drawing the samples
            action = zeros(obj.daction,n);
            count = 1;
            for i = 1 : length(obj.p);
                n = sum(components==i);
                action(:,count:count+n-1) = mymvnrnd(obj.mu(i,:)',obj.Sigma(:,:,i),n);
                count = count + n;
            end
        end
        
        %% GMM.PDF
        function probability = evaluate(obj, Actions)
            probability = zeros(1,size(Actions,2));
            for i = 1 : length(obj.p)
                probability = probability + obj.p(i) * ...
                    exp(loggausspdf(Actions, obj.mu(i,:)', obj.Sigma(:,:,i)));
            end
        end

        %% ENTROPY
        function S = entropy(obj, Actions)
            prob_list = obj.evaluate(Actions);
            idx = isinf(prob_list) | isnan(prob_list) | prob_list == 0;
            prob_list(idx) = 1; % ignore them -> log(1)*1 = 0
            S = -mean(log(prob_list));
        end

        %% GMM.FIT
        function obj = weightedMLUpdate(obj, weights, Actions)
            assert(min(weights >= 0), 'Weights cannot be negative.');
            [~, gmm] = emgm(Actions, obj.gmax, weights);
            obj.p = gmm.ComponentProportion';
            obj.mu = gmm.mu';
            obj.Sigma = gmm.Sigma;
        end

        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.Sigma = obj.Sigma * 1e-100;
        end
        
        function obj = randomize(obj, factor)
            obj.Sigma = obj.Sigma .* factor;
        end
       
    end
    
end
