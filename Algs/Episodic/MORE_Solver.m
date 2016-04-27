classdef MORE_Solver < handle
% Model-based Relative Entropy Stochastic Search.
%
% =========================================================================
% REFERENCE
% A Abdolmaleki, R Lioutikov, N Lau, L P Reis, J Peters, G Neumann
% Model-based Relative Entropy Stochastic Search (2015)
    
    properties
        epsilon       % KL divergence bound
        gamma         % Decreasing factor for the bound on the relative entropy
        entropy_bound % How to choose the minimum entropy constraint ('absolute' vs 'relative', see dual function)
        minH          % Minimum entropy
        b             % Mean of the Gaussian search distribution
        Q             % Covariance of the Gaussian
        Qi            % Inverse of the covariance (store it for faster computation)
        model         % Quadratic reward model
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = MORE_Solver(epsilon, gamma, minH, policy, entropy_bound)
            obj.epsilon = epsilon;
            obj.gamma = gamma;
            obj.minH = minH;
            assert(isa(policy,'GaussianConstant'), 'The search distribution must be a Gaussian with constant mean.')
            obj.Q = policy.Sigma;
            obj.b = policy.mu;
            obj.Qi = inv(obj.Q);
            if nargin < 5, entropy_bound = 'absolute'; end
            obj.entropy_bound = entropy_bound;
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, J, Actions, policy, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            obj.model = quadraticfit(Actions, J, ...
                'weights', W, ...
                'standardize', 0);
            [eta, omega] = obj.optimize;
            divKL = obj.update(eta, omega);
            policy = policy.update(obj.b, obj.Q);
        end
        
        %% CORE
        function [eta, omega] = optimize(obj)
            % Optimization problem settings
            options = optimset('GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 1000 * 5, ...
                'Algorithm', 'interior-point', ...
                'TolX', 10^-8, ...
                'TolFun', 10^-12, ...
                'MaxIter', 1000);
            lowerBound = [1e-8; 1e-8]; % [eta, omega] > 0
            upperBound = [1e8; 1e8];% [eta, omega] < inf
            params = [0.5; 0.5]; % init [eta, omega]
            params = fmincon(@(params)obj.dual(params), ...
                params, [], [], [], [], lowerBound, upperBound, [], options);
            eta = params(1);
            omega = params(2);
        end
        
        %% CLOSED FORM UPDATE
        function divKL = update(obj, eta, omega)
            F_new = inv(eta * obj.Qi - 2 * obj.model.R);
            f_new = eta * (obj.Qi * obj.b) + obj.model.r;
            Q_new = F_new * (eta + omega);
            Qi_new = inv(Q_new);
            b_new = F_new * f_new;
            [~, p] = chol(Q_new);
            if(p ~= 0)
                warning('Covariance is not PSD. Policy has not been updated.');
                divKL = 0;
            else
                divKL = 0.5 * ( ...
                    trace(Q_new \ obj.Q) + (b_new - obj.b)' / Q_new * (b_new - obj.b) ...
                    - length(b_new) + logdet(Q_new,'chol') - logdet(obj.Q,'chol') ...
                    );
                obj.Q = Q_new;
                obj.Qi = Qi_new;
                obj.b = b_new;
            end
        end
        
        %% DUAL FUNCTION
        function [g, gD] = dual(obj, params)
            b = obj.b;
            Q = obj.Q;
            Qi = obj.Qi;
            R = obj.model.R;
            r = obj.model.r;
            dimQ = size(Q,1);

            eta = params(1);
            omega = params(2);
            H = 0.5 * ( dimQ*log(2*pi*exp(1)) + 2*sum(log(diag(chol(Q)))) ); % Current entropy
            
            switch (obj.entropy_bound)
                case 'relative'
                    beta = (obj.gamma * (H - obj.minH) + obj.minH);
                    
                case 'absolute'
                    deltaH = obj.minH - H;
                    if (deltaH > obj.gamma), deltaH = obj.gamma; end
                    if (deltaH < -obj.gamma), deltaH = -obj.gamma; end
                    beta = H + deltaH;
                    
                otherwise
                    error('Unknow entropy bound type.')
            end            

            F = eta * Qi - 2 * R;
            tempQ = ( F / (eta + omega) );
            if any(isnan(tempQ) | any(eig(tempQ) <= 0))
                g = inf;
                gD = inf(2,1);
                warning('Covariance is not PSD.')
                return
            end
            
            f = eta * (Q \ b) + r;
            
            logDetQ = dimQ * log(2*pi) + 2*sum(log(diag(chol(Q))));
            logDetF = dimQ * log((eta+omega)*2*pi) - 2*sum(log(diag(chol(F))));
            
            g = eta * obj.epsilon - omega * beta - 0.5 * eta * (b' * (Q \ b) ) ...
                + 0.5 * f' * (F \ f) - 0.5 * eta * logDetQ + 0.5 * (eta + omega) * logDetF;
            
            % Gradient of dual function (if required by the optimizer)
            if nargout > 1
                Fi = inv(F);
                invF_d_eta = -(Fi' * (Q \ Fi)); % deriv wrt eta of inv(F)
                f_d_eta = Q \ b;                % deriv wrt eta of f
                
                d_eta = obj.epsilon - 0.5 * b' * f_d_eta + 0.5 * f' * invF_d_eta * f ...
                    + f' * ( F \ f_d_eta ) - 0.5 * logDetQ + 0.5 * logDetF ...
                    - 0.5 * (eta + omega) * trace(F \ Qi) + 0.5 * dimQ;
                
                gD = [d_eta
                    -beta + 0.5 * dimQ + 0.5 * logDetF];
            end
        end
        
    end
    
end