classdef gmm_constant
% GMM_CONSTANT Gaussian Mixture Model with constant means and covariances.
% Parameters: means and covariances.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        mu;    % means
        dim;   % length of the action drawn
        Sigma; % covariances
        p;     % mixing proportions
        gmax;  % max number of Gaussians
    end
    
    methods
        
        function obj = gmm_constant(varargin)
            if nargin == 3 % initialize with a single Gaussian
                mu0 = varargin{1};
                sigma0 = varargin{2};
                obj.gmax = varargin{3};
                
                n_params = length(mu0);
                initGauss = gaussian_constant(n_params,mu0,sigma0);
                obj.mu = zeros(obj.gmax,n_params);
                obj.Sigma = zeros(n_params,n_params,obj.gmax);
                for i = 1 : obj.gmax
                    obj.mu(i,:) = initGauss.drawAction;
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
            assert(d1 == d & d2 == d & d3 == k);
            obj.dim = size(obj.mu,2);
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
            assert(min(weights>=0));
            [~, gmm] = emgm(Action, obj.gmax, weights);
            obj.p = gmm.ComponentProportion';
            obj.mu = gmm.mu';
            obj.Sigma = gmm.Sigma;
        end
        
        function obj = randomize(obj, factor)
            obj.Sigma = obj.Sigma .* factor;
        end
       
    end
    
end
