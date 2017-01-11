classdef GmmGibbsConstant < GmmConstant
% GMMGIBBSCONSTANT Gaussian Mixture Model with constant means and 
% covariances. Gating policy is a Gibbs distribution to ensure positiveness 
% and convexity of the weights. Mixing proportions are the weights of the 
% Gibbs distribution.
% Parameters: means, covariances and mixture weights.
% Means are stored in rows, as in Matlab's GMDISTRIBUTION.
%
% As the number of components is variable (up to GMAX), this class does not
% implement the properties THETA and DPARAMS.
    
    methods
        
        %% Constructor
        function obj = GmmGibbsConstant(varargin)
            obj = obj@GmmConstant(varargin{:});
        end
        
        %% GMM.RANDOM
        function action = drawAction(obj,n)
            exponential = exp(obj.p - max(obj.p));
            mix_coef = exponential / sum(exponential);
            components = mymnrnd(mix_coef,n); % Select the Gaussians for drawing the samples
            action = zeros(obj.daction,n);
            count = 1;
            for i = 1 : length(obj.p)
                n = sum(components==i);
                action(:,count:count+n-1) = mymvnrnd(obj.mu(i,:)',obj.Sigma(:,:,i),n);
                count = count + n;
            end
        end
        
        %% GMM.PDF
        function probability = evaluate(obj, Actions)
            exponential = exp(obj.p - max(obj.p));
            mix_coef = exponential / sum(exponential);
            probability = zeros(1,size(Actions,2));
            for i = 1 : length(obj.p)
                probability = probability + mix_coef(i) * ...
                    exp(loggausspdf(Actions, obj.mu(i,:)', obj.Sigma(:,:,i)));
            end
        end
        
        %% GMM.FIT
        function obj = weightedMLUpdate(obj, weights, Actions)
            error('Not supported.')
        end

        %% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            [k,d] = size(obj.mu);

            prob = obj.evaluate(action);
            dlogpdt_mu    = zeros(k*d,1);
            dlogpdt_sigma = zeros(k*d*d,1);
            dlogpdt_mixc  = zeros(k,1);
            
            exponential = exp(obj.p-max(obj.p));

            mix_coef_den = sum(exponential);
            mix_coef = exponential / mix_coef_den;
            
            for i = 1 : k
                idx  = (i-1)*d;
                mu_i = obj.mu(i,:)';
                C_i  = obj.Sigma(:,:,i);
                dens = mvnpdf(action, mu_i, C_i);
                
                % Compute gradient w.r.t. mean
                grad_mu = C_i \ (action - mu_i);
                dlogpdt_mu(1+idx:idx+d,1) = dens * grad_mu * mix_coef(i);
                
                % Compute gradient w.r.t. covariance
                tmp = inv(C_i)';
                A = -0.5 * tmp;
                B =  0.5 * tmp * (action - mu_i) * (action - mu_i)' * tmp;
                grad_C = A + B;
                dlogpdt_sigma(1+idx*d:idx*d+d^2,1) = dens * grad_C(:) * mix_coef(i);
                
                % Compute gradient w.r.t. mixing coefficients
                dlogpdt_mixc(i) = 0;
                for j = 1 : k
                    mu_j = obj.mu(j,:)';
                    C_j  = obj.Sigma(:,:,j);
                    dens_j = mvnpdf(action, mu_j, C_j);
                    if (i == j)
                        dlogpdt_mixc(i) = dlogpdt_mixc(i) + dens_j * ...
                            exp(obj.p(i)-max(obj.p)) * (mix_coef_den - exp(obj.p(i)-max(obj.p))) / mix_coef_den^2;
                    else
                        dlogpdt_mixc(i) = dlogpdt_mixc(i) - dens_j * ...
                            exp(obj.p(i)-max(obj.p)) * exp(obj.p(j)-max(obj.p)) / mix_coef_den^2;
                    end
                end
            end
            
            dlogpdt = [dlogpdt_mu; dlogpdt_sigma; dlogpdt_mixc];
            dlogpdt = dlogpdt / prob;
        end
        
        %% Gradient update
        function obj = update(obj, direction)
            [k,d] = size(obj.mu);
            for i = 1:k
                idx = (i-1)*d;
                mu_i = direction(1+idx:idx+d,1);
                obj.mu(i,:) = obj.mu(i,:) + mu_i';
                sigma_i = direction(k*d+1+idx*d:k*d+idx*d+d^2,1);
                obj.Sigma(:,:,i) = eye(d,d)*1e-6 + ...
                    nearestSPD(obj.Sigma(:,:,i) + reshape(sigma_i,d,d));
            end
            obj.p = obj.p + direction(end-k+1:end,1);
        end
        
    end
    
end
