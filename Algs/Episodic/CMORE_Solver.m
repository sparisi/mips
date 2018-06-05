classdef CMORE_Solver < handle
% Contextual Model-based Relative Entropy Stochastic Search.
% It supports Importance Sampling (IS).
%
% =========================================================================
% REFERENCE
% V Tangkaratt, H van Hoof, S Parisi, M Sugiyama, G Neumann, J Peters
% Policy Search with High-Dimensional Context Variables (2017)
    
    properties
        epsilon       % KL divergence bound
        gamma         % Decreasing factor for the bound on the relative entropy
        entropy_bound % How to choose the minimum entropy constraint ('absolute' vs 'relative', see dual function)
        minH          % Minimum entropy
        b             % Bias of the mean of the Gaussian search distribution
        K             % Linear term of the mean of the Gaussian search distribution
        Q             % Covariance of the Gaussian
        Qi            % Inverse of the covariance (store it for faster computation)
        model         % Quadratic reward model
        eta           % Current Lagrangian
        omega         % Current Lagrangian
        lambda_l1     % L1-norm regularizer
        lambda_l2     % L2-norm regularizer
        lambda_nn     % Nuclear norm regularizer
        std_data_flag % 1 to standardize data when fitting the quadratic model
        alg_name      % Name of the proximal gradient algorithm for fitting the quadratic model
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = CMORE_Solver(epsilon, gamma, minH, policy, entropy_bound)
            obj.epsilon = epsilon;
            obj.gamma = gamma;
            obj.minH = minH;
            assert(isa(policy,'GaussianLinear'), 'The search distribution must be a Gaussian with mean linearly depending on some features.')
            obj.Q = policy.Sigma;
            obj.b = policy.A(:,1);
            obj.K = policy.A(:,2:end);
            obj.Qi = inv(obj.Q);
            if nargin < 5, entropy_bound = 'absolute'; end
            obj.entropy_bound = entropy_bound;
            obj.eta = 1e3;
            obj.omega = 0.5;
            obj.lambda_l1 = 0;
            obj.lambda_l2 = 0;
            obj.lambda_nn = 0;
            obj.std_data_flag = 0;
            obj.model.H = [];
            obj.model.R1 = [];
            obj.model.R2 = [];
            obj.model.Rc = [];
            obj.model.r1 = [];
            obj.model.r2 = [];
            obj.model.r0 = [];
            obj.alg_name = 'proxgrad';
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, J, Actions, Phi, policy, W)
            if nargin < 6, W = ones(1, size(J,2)); end % IS weights
            obj.model = quadraticfit2(Actions, Phi, J, [], ...
                'weights', W, 'standardize', obj.std_data_flag, ...
                'lambda_l1', obj.lambda_l1, ...
                'lambda_l2', obj.lambda_l2, ...
                'lambda_nn', obj.lambda_nn, ...
                'alg', obj.alg_name);
            obj.optimize(Phi);
            divKL = obj.update(Phi);
            policy = policy.update([obj.b, obj.K], obj.Q);
        end
        
        %% CORE
        function optimize(obj, Phi)
            % Optimization problem settings
            options = optimset('GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 5000, ...
                'Algorithm', 'interior-point', ...
                'TolX', 10^-8, ...
                'TolFun', 10^-12, ...
                'MaxIter', 1000);
            lowerBound = [1e-8; 1e-8]; % [eta, omega] > 0
            upperBound = [1e8; 1e8];% [eta, omega] < inf
            params = [obj.eta; obj.omega]; % init [eta, omega]
            params = fmincon(@(params)obj.dual(params, Phi), ...
                params, [], [], [], [], lowerBound, upperBound, [], options);
            obj.eta = params(1);
            obj.omega = params(2);
        end

        %% CLOSED FORM UPDATE
        function divKL = update(obj, Phi)
            F_new = inv(obj.eta * obj.Qi - 2 * obj.model.R1);
            L_new = obj.eta * (obj.Qi * obj.K) + 2 * obj.model.Rc;
            f_new = obj.eta * (obj.Qi * obj.b) + obj.model.r1;
            Q_new = F_new * (obj.eta + obj.omega);
            Qi_new = inv(Q_new);
            b_new = F_new * f_new;
            [~, flag] = chol(Q_new);
            if(flag ~= 0)
                warning('Covariance is not PSD. Policy has not been updated.');
                divKL = 0;
            else
                q.A = [obj.b, obj.K];
                q.Sigma = obj.Q;
                obj.Q = Q_new;
                obj.Qi = Qi_new;
                obj.b = b_new;
                obj.K = F_new * L_new;
                p.A = [obj.b, obj.K];
                p.Sigma = obj.Q;
                divKL = kl_mvn2(p,q,Phi);
            end
        end
        
        %% DUAL FUNCTION
        function [g, gD] = dual(obj, params, Phi)
            b = obj.b;
            K = obj.K;
            Q = obj.Q;
            Qi = obj.Qi;
            Raa = obj.model.R1;
%             Rss = obj.model.R2;
            Ras = obj.model.Rc;
            ra = obj.model.r1;
%             rs = obj.model.r2;
            dimQ = size(Q,1);
            N = size(Phi,2);

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
            
            F = eta * Qi - 2 * Raa;
            tempQ = ( F / (eta + omega) );
            if any(isnan(tempQ) | any(eig(tempQ) <= 0))
                g = nan;
                gD = nan(2,1);
                warning('Covariance is not PSD.')
                return
            end
            
            f = eta * (Qi * b) + ra;
            L = eta * (Qi * K) + 2 * Ras;
            invF = inv(F);
            M = 0.5 * ( L' * (invF * L) - eta * K' * (Qi * K) );
            
            logDetQ = dimQ * log(2*pi) + 2*sum(log(diag(chol(Q))));
            logDetF = dimQ * log((eta+omega)*2*pi) - 2*sum(log(diag(chol(F))));
            
            g_constant = eta * obj.epsilon - omega * beta - 0.5 * eta * (b' * (Q \ b) ) ...
                + 0.5 * f' * (invF * f) - 0.5 * eta * logDetQ + 0.5 * (eta + omega) * logDetF;
            g_linear = sum(Phi' * (L' * (invF * f) - eta * K' * (Qi * b))) / N;
            g_squared = sum( sum( (Phi'*M)' .* Phi) ) / N;
            
            g = g_squared + g_linear + g_constant;

            % Gradient of dual function (if required by the optimizer)
            if nargout > 1
                invF_d_eta = -(invF' * (Q \ invF)); % deriv wrt eta of inv(F)
                f_d_eta = Q \ b;                    % deriv wrt eta of f
                
                d_eta_constant = obj.epsilon - 0.5 * b' * f_d_eta + 0.5 * f' * invF_d_eta * f ...
                    + f' * ( invF * f_d_eta ) - 0.5 * logDetQ + 0.5 * logDetF ...
                    - 0.5 * (eta + omega) * trace(invF * Qi) + 0.5 * dimQ;
                d_eta_linear = L' * (invF * f_d_eta) - L' * ((invF'*(Q\(invF*f)))) + (Q\K)'*(invF*f) - K'*(Q\b);
                d_eta_squared = 0.5*L'*(invF_d_eta*L) + (Q\K)'*(invF*L) - 0.5*K'*Qi*K;

                gD = [d_eta_constant + sum(Phi' * d_eta_linear) / N + sum(sum((Phi'*d_eta_squared)'.*Phi)) / N;
                    -beta + 0.5 * dimQ + 0.5 * logDetF];
            end
        end
        
        %% GET RETURN APPROXIMATION
        function R = getR(obj, action, context)
            R = obj.model.eval(action, context);
        end

        %% PLOTTING
        function plotR(obj, actionLB, actionUB, contextLB, contextUB)
            if length(contextLB) > 1 || length(actionLB) > 1, return, end

            n = 30;
            x = linspace(actionLB,actionUB,n);
            y = linspace(contextLB,contextUB,n);
            [X, Y] = meshgrid(x,y);
            R = obj.getR(X(:)',Y(:)');
            updatesurf('Return Model', X, Y, reshape(R,n,n))
        end        
        
    end
    
end
