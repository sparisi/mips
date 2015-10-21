classdef (Abstract) policy_gaussian < policy
% POLICY_GAUSSIAN Generic class for Gaussian policies, both with constant
% and state-dependent parameters.
    
    methods (Abstract)
        getParams(obj)
    end
    
    methods
        
        % In all methods, STATES are matrices S-by-N, where S is the size
        % of one state and N is the number of states.
        % Similarly, ACTIONS are matricies A-by-N.
        
        %%% Logarithm of probability density function
        function logprob = logpdf(obj, varargin)
            if numel(varargin) == 2 % linear mean
                States = varargin{1};
                Actions = varargin{2};
                ns = size(States,2);
                na = size(Actions,2);
                assert(ns == na ... % logprob(i) = prob of drawing Actions(i) in States(i)
                    || na == 1 ... % logprob(i) = prob of drawing a single action in States(i)
                    || ns == 1); % logprob(i) = prob of drawing Actions(i) in a single state
                params = obj.getParams(States);

            elseif numel(varargin) == 1 % constant mean
                params = obj.getParams();
                Actions = varargin{1};

            else
                error('Wrong number of input arguments.')
                
            end
            
            try
                U = params.cholU;
            catch
                Sigma = params.Sigma;
                U = chol(Sigma);
            end
            mu = params.mu;
            Actions = bsxfun(@minus, Actions, mu);
            d = size(Actions,1);
            Q = U' \ Actions;
            q = dot(Q,Q,1); % quadratic term (M distance)
            c = d * log(2*pi) + 2 * sum(log(diag(U))); % normalization constant
            logprob = -(c+q)/2;
        end
        
        %%% MVNPDF
        function probability = evaluate(obj, varargin)
            probability = exp(obj.logpdf(varargin{:}));
        end
        
        %%% MVNRND
        function action = drawAction(obj, varargin)
            if numel(varargin) == 1 % linear mean
                States = varargin{1};
                n = size(States,2); % draw N samples, one for each state
                params = obj.getParams(States);

            elseif numel(varargin) == 0 % constant mean
                n = 1; % draw one sample
                params = obj.getParams;
            
            else
                error('Wrong number of input arguments.')
            
            end
            
            try
                U = params.cholU;
            catch
                Sigma = params.Sigma;
                U = chol(Sigma);
            end
            mu = params.mu;
            action = U'*randn(obj.dim,n) + mu;
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, varargin)
            params = obj.getParams;
            Sigma = params.Sigma;
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(Sigma) );
        end

        %%% Zero variance
        function obj = makeDeterministic(obj)
            obj.theta(end-obj.dim_explore+1:end) = 1e-8;
        end

        %%% Increase variance by factor
        function obj = randomize(obj, factor)
            obj.theta(end-obj.dim_explore+1:end) = obj.theta(end-obj.dim_explore+1:end) .* factor;
        end
        
        %% PLOTTING
        %%% Plot N(mu(i),Sigma(i,i)) for each dimension i (NORMPDF)
        function plotPDF(obj, varargin)
            if numel(varargin) == 1 % linear mean
                States = varargin{1};
                assert(size(States,2) == 1) % plot only for one state
                params = obj.getParams(States);
                
            elseif numel(varargin) == 0 % constant mean
                params = obj.getParams;

            else
                error('Wrong number of input arguments.')
            
            end
            
            Sigma = params.Sigma;
            mu = params.mu;
            figure; hold all
            xlabel 'x_i'
            ylabel 'Policy density'
            x = max(abs(mu)) + 2*max(abs(Sigma(:)));
            range = -x : 0.1 : x;
            for i = 1 : length(mu)
                norm = exp(-0.5 * ((range - mu(i))./Sigma(i,i)).^2) ./ (sqrt(2*pi) .* Sigma(i,i));
                plot(range, norm)
            end
        end
        
    end

end
