%%% Gaussian with linear mean and constant diagonal covariance: N(K*phi,S).
%%% Params: mean and diagonal std (S = diag(s)^2).
classdef gaussian_diag_linear < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    methods
        
        function obj = gaussian_diag_linear(basis, dim, init_k, init_sigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(init_k,2))
            assert(dim == size(init_k,1))
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == 1)

            obj.basis = basis;
            obj.dim = dim;
            obj.theta = [init_k(:); init_sigma];
            obj.dim_explore = length(init_sigma);
        end
        
        function probability = evaluate(obj, state, action)
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = diag(obj.theta(n_k+1:end));
            probability = mvnpdf(action, mu, sigma.^2);
        end
        
        function action = drawAction(obj, state)
            phi = feval(obj.basis,state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = diag(obj.theta(n_k+1:end));
            action = mvnrnd(mu,sigma.^2)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, state)
            n_k = obj.dim*feval(obj.basis);
            sigma = diag(obj.theta(n_k+1:end));
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma.^2) );
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlogpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = obj.theta(n_k+1:end);

            dlogpdt_k = sigma.^-2 .* (action - mu) * phi';
            dlogpdt_sigma = -sigma.^-1 + (action - mu).^2 ./ sigma.^3;

            dlogpdt = [dlogpdt_k(:); dlogpdt_sigma];
        end
        
        %%% Hessian of the logarithm of the policy
        function hlogpdt = hlogPidtheta(obj, state, action)
            if (nargin == 1)
                hlogpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis);
            K = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = K*phi;
            sigma = obj.theta(n_k+1:end);
            invsigma = diag(sigma.^-2);
            phimat = kron(eye(obj.dim),phi');
            diff = action - mu;
            
            dm = feval(obj.basis);
            ds = obj.dim_explore;
            hlogpdt = zeros(length(obj.theta));
            
            % dlogpdt / (dmu dmu)
            hlogpdt(1:dm,1:dm) = - phimat' * invsigma * phimat;

            % dlogpdt / (dsigma dsigma)
            hlogpdt(dm+1:dm+ds,dm+1:dm+ds) = invsigma - 3.0 * diff.^2 * sigma.^-4;
            
            % dlogpdt / (dmu dsigma)
            hlogpdt(dm+1:dm+ds, 1:dm) = - 2 * phimat * diff * sigma.^-3;
            hlogpdt(1:dm, dm+1:dm+ds) = hlogpdt(dm+1:dm+ds, 1:dm);
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            n_k = obj.dim*feval(obj.basis);
            obj.theta(n_k+1:end) = 1e-8;
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            Sigma = zeros(obj.dim);
            D = diag(weights);
            N = size(Action,1);
            W = (Phi' * D * Phi + 1e-8 * eye(size(Phi,2))) \ Phi' * D * Action;
            W = W';
            for k = 1 : N
                Sigma = Sigma + (weights(k) * ( Action(k,:)' - W*Phi(k,:)') ...
                    * (Action(k,:)' - W*Phi(k,:)')' );
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = diag(diag(Sigma));
            obj.theta = [W(:); diag(sqrt(Sigma))];
        end
        
        function params = getParams(obj)
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            sigma = diag(obj.theta(n_k+1:end));
            
            params.A = k;
            params.Sigma = sigma.^2;
        end
        
        function obj = randomize(obj, factor)
            n_k = obj.dim*feval(obj.basis);
            obj.theta(n_k+1:end) = obj.theta(n_k+1:end) .* factor;
        end
        
    end
    
end
