classdef (Abstract) GaussianLinear < Gaussian
% GAUSSIANLINEAR Generic class for Gaussian policies. Its mean linearly 
% depends on some features, i.e., mu = A*[1; phi].
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        basis % Basis functions
        A     % Mean linear term
        Sigma % Covariance
        U     % Cholesky decomposition
    end

    methods

        %% Adds the constant feature 1 to the basis function
        function phi1 = basis1(obj, States)
            phi1 = [ones(1,size(States,2)); obj.basis(States)];
        end
        
        %% Derivative of the logarithm of the policy
        function logprob = logpdf(obj, Actions, States)
            ns = size(States,2);
            na = size(Actions,2);
            assert(ns == na ... % logprob(i) = prob of drawing Actions(i) in States(i)
                || na == 1 ... % logprob(i) = prob of drawing a single action in States(i)
                || ns == 1, ... % logprob(i) = prob of drawing Actions(i) in a single state
                'Number of states and actions is not consistent.')
            phi = obj.basis1(States);
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
                phi = obj.basis1(States);
                mu = obj.A*phi;
                Actions = obj.U'*randn(obj.daction,n) + mu;
            else
                assert(d == n, 'Number of states and policies is not consistent.')
                assert(isequal(obj.basis), 'All policies must have the same basis functions.')
                phi = obj(1).basis1(States);
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
            
            phi = obj.basis1(state);
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
