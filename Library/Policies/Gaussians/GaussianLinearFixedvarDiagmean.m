classdef GaussianLinearFixedvarDiagmean < GaussianLinear
% GAUSSIANLINEARFIXEDVARDIAGMEAN Gaussian distribution with linear diagonal
% mean and fixed covariance: N(A*phi,S), A diagonal matrix.
% Parameters: mean A.
    
    methods
        
        function obj = GaussianLinearFixedvarDiagmean(basis, dim, initA, Sigma)
            assert(isscalar(dim) && ...
                basis() == dim && ...
                size(initA,2) == dim && ...
                size(initA,1) == dim && ...
                size(Sigma,1) == dim && ...
                size(Sigma,2) == dim, ...
                'Dimensions are not consistent.')
            [initU, p] = chol(Sigma);
            assert(p == 0, 'Covariance must be positive definite.')

            obj.daction = dim;
            obj.theta = diag(initA);
            obj.basis = basis;
            obj.dparams = length(obj.theta);
            obj.Sigma = Sigma;
            obj.U = initU;
            obj.A = initA;
        end
        
        %% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            phi = obj.basis(state);
            dlpdt = bsxfun(@times, (obj.Sigma) \ (action - obj.A*phi), phi);
        end
        
        %% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            phi = obj.basis(state);
            invSigma = inv(obj.Sigma);
            hlpdt = bsxfun(@times, -bsxfun(@times,permute(phi,[3 1 2]),permute(phi,[1 3 2])), invSigma);
        end

        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            obj.A = diag(obj.theta);
        end

        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.Sigma = 1e-8 * obj.Sigma;
            obj.U = chol(obj.Sigma);
        end
        
        function obj = randomize(obj,varargin)
            warning('This policy should not be randomized!')
        end
        
        %% ============================================================= %%
        %  OVERRIDE Gaussian function, as there is no constant feature    %
        %  =============================================================  %
        %% Derivative of the logarithm of the policy
        function logprob = logpdf(obj, Actions, States)
            ns = size(States,2);
            na = size(Actions,2);
            assert(ns == na ... % logprob(i) = prob of drawing Actions(i) in States(i)
                || na == 1 ... % logprob(i) = prob of drawing a single action in States(i)
                || ns == 1, ... % logprob(i) = prob of drawing Actions(i) in a single state
                'Number of states and actions is not consistent.')
            phi = obj.basis(States);
            mu = obj.A*phi;
            Actions = bsxfun(@minus, Actions, mu);
            d = size(Actions,1);
            Q = obj.U' \ Actions;
            q = dot(Q,Q,1); % quadratic term (M distance)
            c = d * log(2*pi) + 2 * sum(log(diag(obj.U))); % normalization constant
            logprob = -(c+q)/2;
        end
        
        %% MVNPDF
        function probability = evaluate(obj, varargin)
            probability = exp(obj.logpdf(varargin{:}));
        end
        
        %% MVNRND
        function Actions = drawAction(obj, States)
            d = numel(obj);
            n = size(States,2); % draw N samples, one for each state
            if d == 1
                phi = obj.basis(States);
                mu = obj.A*phi;
                Actions = obj.U'*randn(obj.daction,n) + mu;
            else
                assert(d == n, 'Number of states and policies is not consistent.')
                assert(isequal(obj.basis), 'All policies must have the same basis functions.')
                phi = obj(1).basis(States); 
                Actions = multimvnrnd(cat(3,obj.U),cat(3,obj.A),phi);
            end
        end
        
        %% PLOT PDF
        function fig = plot(obj, state)
        % Plot N(mu(i),Sigma(i,i)) for each dimension i (NORMPDF)
            ns = size(state,2);
            assert(ns == 1, 'PDF can be plotted only for one state.')
            fig = figure(); hold all
            xlabel 'x'
            ylabel 'pdf(x)'
            
            phi = obj.basis(state);
            mu = obj.A*phi;
            range = ndlinspace(mu - 3*diag(obj.Sigma), mu + 3*diag(obj.Sigma), 100);
            
            norm = bsxfun(@times, exp(-0.5 * bsxfun(@times, ...
                bsxfun(@minus,range,mu), 1./diag(obj.Sigma)).^2), ... 
                1./(sqrt(2*pi) .* diag(obj.Sigma)));
            plot(range', norm')
            legend show
        end                
        
    end
    
end
