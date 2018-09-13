classdef REPSep_constrained_Solver < handle
% Like REPSep_Solver, but the maximum likelihood update is constrained to
% bound the KL diverngence and the entropy of the new policy as well.
%
% =========================================================================
% REFERENCE
% A Abdolmaleki, B Price, N Lau, L P Reis, G Neumann
% Deriving and Improving CMA-ES with Information Geometric Trust Regions
% (2017)
%
% Note: the paper only constrains on the KL divergence, while this
% implementation also constrains on the entropy. 

    properties
        epsilon  % KL divergence bound
        eta_reps % Lagrangian
        eta_pi   % Lagrangian for the maximum likelihood update (KL divergence)
        omega_pi % Lagrangian for the maximum likelihood update (entropy)
        kappa    % H(pi_old) - kappa <= H(pi_new)
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = REPSep_constrained_Solver(epsilon, kappa)
            obj.epsilon = epsilon;
            obj.eta_reps = 1e3;
            obj.eta_pi = 1e3;
            obj.omega_pi = 1e3;
            obj.kappa = kappa;
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, J, Actions, policy, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            
            % Get REPS weights
            [d, divKL] = obj.optimize(J, W);
            
            % Optimize ML dual
            options = optimoptions(@fmincon, 'Algorithm', 'interior-point', ...
                'GradObj', 'off', ...
                'Display', 'off', ...
                'MaxFunEvals', 5000, ...
                'TolX', 10^-8, ...
                'TolFun', 10^-12, ...
                'MaxIter', 1000);
            lowerBound = [1e-8; 1e-8]; % [eta, omega] > 0
            upperBound = [1e8; 1e8];% [eta, omega] < inf
            params = fmincon(@(params)obj.dual_wml(params, policy.mu, policy.Sigma, Actions, d), ...
                [obj.eta_pi; obj.omega_pi], [], [], [], [], lowerBound, upperBound, [], options);
            obj.eta_pi = params(1);
            obj.omega_pi = params(2);

            % Weighted constrained ML update
            [~, mu_new, sigma_new] = obj.dual_wml(params, policy.mu, policy.Sigma, Actions, d);
            policy = policy.update(mu_new, sigma_new);
        end

        %% CORE
        function [d, divKL] = optimize(obj, J, W)
            if nargin < 3, W = ones(1,size(J,2)); end % IS weights
            
            % Optimization problem settings
            options = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 500, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

            obj.eta_reps = fmincon(@(eta)obj.dual_reps(eta,J,W), ...
                obj.eta_reps, [], [], [], [], 1e-8, 1e8, [], options);
            
            % Compute weights for weighted ML update
            d = W .* exp( (J - max(J)) / obj.eta_reps );

            % Compute KL divergence
            qWeighting = W;
            pWeighting = d;
            divKL = kl_mle(pWeighting, qWeighting);
        end
        
        %% DUAL FUNCTION (REPS)
        function [g, gd, h] = dual_reps(obj, eta, J, W)
            if nargin < 4, W = ones(1,size(J,2)); end % IS weights
            n = sum(W);
            
            maxR = max(J);
            weights = W .* exp( ( J - maxR ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsR = sum( weights .* (J - maxR) );
            sumWeightsRR = sum( weights .* (J - maxR).^2 );

            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxR; % dual
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsR / (eta * sumWeights); % gradient
            h = ( sumWeightsRR * sumWeights - sumWeightsR^2 ) / ( eta^3 * sumWeights^2 ); % hessian
        end
        
        %% DUAL FUNCTION (MAXIMUM LIKELIHOOD)
        function [g, mu_new, sigma_new] = dual_wml(obj, params, mu, sigma, x, w)
            eta = params(1);
            omega = params(2);
            
            [d, n] = size(x);
            
            c = d*log(2*pi);
            H = 0.5*logdet(sigma,'chol') + c/2 + d/2;
            beta = H - obj.kappa;
            W = sum(w);
            mu_new = (x*w' + eta*mu) / (W + eta);
            diff = bsxfun(@minus, x, mu_new);
            sigma_s = diff*diag(w)*diff';
            
            sigma_new = (sigma_s + eta*sigma + eta*(mu_new-mu)*(mu_new-mu)') / (W + eta - omega);
            tmp = sum(sum((sigma_new\eye(d)).*conj(bsxfun(@times,diff,w)*diff'),2));
            
            g = - 0.5*tmp ...
                - 0.5*eta*trace(sigma_new \ sigma) ...
                - 0.5*eta*trace(sigma_new \ (mu_new - mu) * (mu_new - mu)') ...
                + 0.5*(omega - eta - W)*logdet(sigma_new,'chol') ...
                + 0.5*eta*(d + logdet(sigma,'chol') + 2*obj.epsilon) ...
                + 0.5*omega*(c + d - 2*beta) ...
                - 0.5*W*c;
        end
        
    end
    
end
