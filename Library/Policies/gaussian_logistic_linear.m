%%% Gaussian with linear mean and logistic covariance: N(K*phi,S).
%%% Params: mean and logistic weights (S = tau/(1+exp(-w)).
classdef gaussian_logistic_linear < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end

    properties(GetAccess = 'public', SetAccess = 'public')
        tau;
    end
    
    methods
        
        function obj = ...
                gaussian_logistic_linear(basis, dim, init_k, init_sigma_w, max_variance)
            assert(isscalar(dim))
            assert(feval(basis) == size(init_k,2))
            assert(dim == size(init_k,1))
            assert(size(init_sigma_w,1) == dim)
            assert(size(init_sigma_w,2) == 1)
            assert(size(max_variance,1) == size(init_sigma_w,1));
            assert(size(max_variance,2) == size(init_sigma_w,2));

            obj.theta = [init_k(:); init_sigma_w];
            obj.basis = basis;
            obj.dim   = dim;
            obj.tau   = max_variance;
            obj.dim_explore = length(init_sigma_w);
        end
        
        function probability = evaluate(obj, state, action)
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            
            % Compute mean vector
            phi   = feval(obj.basis, state);
            k     = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            MU    = k * phi;
            probability = mvnpdf(action, MU, SIGMA);
        end
        
        function action = drawAction(obj, state)
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            
            % Compute mean vector
            phi    = feval(obj.basis, state);
            k      = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            MU     = k * phi;
            action = mvnrnd(MU, SIGMA)';
        end

        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlpdt = size(obj.theta,1);
                return
            end
            
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA_INV = diag(1./logv);
            
            % Compute mean vector
            phi   = feval(obj.basis, state);
            k     = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            MU    = k * phi;
            
            dlpdt = zeros(size(obj.theta));
            
            dmu = SIGMA_INV * (action - MU) * phi';
            dlpdt(1:end-obj.dim) = dmu(:);
            
            for i = 1 : obj.dim
                wi = obj.theta(end-obj.dim+i);
                A = -0.5 * exp(-wi) / (1 + exp(-wi));
                B = 0.5 * exp(-wi) / obj.tau(i) * (action(i) - MU(i))^2;
                dlpdt(end-obj.dim+i) = A + B;
            end
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, state)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(SIGMA) );
        end
        
        function obj = makeDeterministic(obj)
            obj.tau = 1e-8 * ones(size(obj.tau));
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function params = getParams(obj)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            sigma = diag(logv);
            k     = vec2mat(obj.theta(1:end-obj.dim),obj.dim);

            params.A = k;
            params.Sigma = sigma;
        end
        
        function obj = randomize(obj, factor)
            obj.theta(end-obj.dim+1:end) = obj.theta(end-obj.dim+1:end) .* factor;
        end
        
        function areEq = eq(obj1, obj2)
            areEq = eq@policy(obj1,obj2);
            if max(areEq)
                areEqTau = bsxfun( @and, [obj1(:).tau], [obj2(:).tau] );
                if size(areEq,1) ~= size(areEqTau,1)
                    areEqTau = areEqTau';
                end
                areEq = bitand( areEq, areEqTau);
            else
                return;
            end
        end
        
    end
    
end
