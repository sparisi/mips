classdef GmmLinear < Policy
% GMMLINEAR Gaussian Mixture Model with linear means and constant covariances.
% Parameters: means, covariances and mixture weights.
% Means are stored in rows, as in Matlab's GMDISTRIBUTION.
%
% As the number of components is variable (up to GMAX), this class does not
% implement the properties THETA and DPARAMS.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis % basis functions
        A     % means coefficients
        Sigma % covariances
        p     % mixing proportions
        gmax  % max number of Gaussians
    end
    
    methods
        
        %% Constructor
        function obj = GmmLinear(varargin)
            if nargin == 4 % initialize with a single Gaussian
                obj.basis = varargin{1};
                A0 = varargin{2};
                Sigma0 = varargin{3};
                obj.gmax = varargin{4};
                for i = 1 : obj.gmax
                    obj.A(:,:,i) = A0;
                    obj.Sigma(:,:,i) = Sigma0;
                end
                obj.p = ones(1,obj.gmax) / obj.gmax;
            elseif nargin == 5 % directly init with all components
                obj.basis = varargin{1};
                obj.A = varargin{2};
                obj.Sigma = varargin{3};
                obj.gmax = varargin{4};
                obj.p = varargin{5};
            else
                error('Wrong number of input arguments.')
            end
            [d,m,k] = size(obj.A);
            [d1,d2,d3] = size(obj.Sigma);
            assert(d1 == d & d2 == d & d3 == k);
            assert(m == obj.basis()+1);
            obj.daction = size(obj.A,1);
            obj.no_bias = false;
        end
        
        %% GMM.RANDOM
        function action = drawAction(obj, States)
            n = size(States,2);
            components = mymnrnd(obj.p,n); % Select the Gaussians for drawing the samples
            action = zeros(obj.daction,n);
            phi = obj.get_basis(States);
            for i = 1 : length(obj.p)
                idx = components == i;
                action(:,idx) = mymvnrnd(obj.A(:,:,i) * phi(:,idx),obj.Sigma(:,:,i));
            end
        end
        
        %% GMM.PDF
        function probability = evaluate(obj, States, Actions)
            probability = zeros(1,size(Actions,2));
            phi = obj.get_basis(States);
            for i = 1 : length(obj.p)
                probability = probability + obj.p(i) * ...
                    exp(loggausspdf(Actions, obj.A(:,:,i) * phi, obj.Sigma(:,:,i)));
            end
        end
        
        %% Entropy
        function S = entropy(obj, States, Actions)
            prob_list = obj.evaluate(States, Actions);
            idx = isinf(prob_list) | isnan(prob_list) | prob_list == 0;
            prob_list(idx) = 1; % ignore them -> log(1)*1 = 0
            S = -mean(log(prob_list));
        end
        
        %% GMM.FIT
        function obj = weightedMLUpdate(obj, weights, Actions, Phi)
            assert(min(weights>=0));
            [~, gmm] = emgm_linear(Actions, Phi, obj.gmax, weights);
            obj.p = gmm.ComponentProportion;
            obj.A = gmm.A;
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
