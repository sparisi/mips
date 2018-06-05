classdef MORE2_Solver < handle
% Model-based Relative Entropy Stochastic Search without entropy
% constraint.
% This is equivalent of REPS with a quadratic model to approximate the
% return and to fit a Gaussian policy in closed form.
%
% =========================================================================
% REFERENCE
% A Abdolmaleki, R Lioutikov, N Lau, L P Reis, J Peters, G Neumann
% Model-based Relative Entropy Stochastic Search (2015)
    
    properties
        epsilon       % KL divergence bound
        b             % Mean of the Gaussian search distribution
        Q             % Covariance of the Gaussian
        Qi            % Inverse of the covariance (store it for faster computation)
        model         % Quadratic reward model
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = MORE2_Solver(epsilon, policy)
            obj.epsilon = epsilon;
            assert(isa(policy,'GaussianConstant'), 'The search distribution must be a Gaussian with constant mean.')
            obj.Q = policy.Sigma;
            obj.b = policy.mu;
            obj.Qi = inv(obj.Q);
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, J, Actions, policy, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            obj.model = quadraticfit(Actions, J, ...
                'weights', W, ...
                'standardize', 0);
            eta = obj.optimize;
            divKL = obj.update(eta);
            policy = policy.update(obj.b, obj.Q);
        end
        
        %% CORE
        function eta = optimize(obj)
            % Optimization problem settings
            options = optimset('GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 1000 * 5, ...
                'Algorithm', 'interior-point', ...
                'TolX', 10^-8, ...
                'TolFun', 10^-12, ...
                'MaxIter', 1000);
            lowerBound = 1e-8; % eta > 0
            upperBound = 1e8; % eta < inf
            eta = 0.5; % init eta
            eta = fmincon(@(eta)obj.dual(eta), ...
                eta, [], [], [], [], lowerBound, upperBound, [], options);
        end
        
        %% CLOSED FORM UPDATE
        function divKL = update(obj, eta)
            F_new = inv(eta * obj.Qi - 2 * obj.model.R);
            f_new = eta * (obj.Qi * obj.b) + obj.model.r;
            Q_new = F_new * eta;
            Qi_new = inv(Q_new);
            b_new = F_new * f_new;
            [~, p] = chol(Q_new);
            if(p ~= 0)
                warning('Covariance is not PSD. Policy has not been updated.');
                divKL = 0;
            else
                divKL = 0.5 * ( ...
                    trace(obj.Qi * Q_new) + (obj.b - b_new)' * obj.Qi * (obj.b - b_new) ...
                    - length(b_new) + logdet(obj.Q,'chol') - logdet(Q_new,'chol') ...
                    );
                obj.Q = Q_new;
                obj.Qi = Qi_new;
                obj.b = b_new;
            end
        end
        
        %% DUAL FUNCTION
        function [g, gD] = dual(obj, eta)
            b = obj.b;
            Q = obj.Q;
            Qi = obj.Qi;
            R = obj.model.R;
            r = obj.model.r;
            dimQ = size(Q,1);

            F = eta * Qi - 2 * R;
            tempQ = ( F / eta );
            if any(any(isnan(tempQ) | any(eig(tempQ) <= 0)))
                g = nan;
                gD = nan(2,1);
                warning('Covariance is not PSD.')
                return
            end
            
            Qinv_b = Q \ b;
            f = eta * Qinv_b + r;
            
            logDetQ = dimQ * log(2*pi) + 2*sum(log(diag(chol(Q))));
            logDetF = dimQ * log(eta*2*pi) - 2*sum(log(diag(chol(F))));
            
            g = eta * obj.epsilon - 0.5 * eta * (b' * Qinv_b ) ...
                + 0.5 * f' * (F \ f) - 0.5 * eta * logDetQ + 0.5 * eta * logDetF;
            
            % Gradient of dual function (if required by the optimizer)
            if nargout > 1
                Fi = inv(F);
                invF_d_eta = -(Fi' * (Q \ Fi)); % deriv wrt eta of inv(F)
                f_d_eta = Qinv_b;               % deriv wrt eta of f
                
                gD = obj.epsilon - 0.5 * b' * f_d_eta + 0.5 * f' * invF_d_eta * f ...
                    + f' * ( Fi * f_d_eta ) - 0.5 * logDetQ + 0.5 * logDetF ...
                    - 0.5 * eta * trace(Fi * Qi) + 0.5 * dimQ;
            end
        end
        
    end
    
end
