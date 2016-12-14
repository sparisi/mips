classdef NES_Solver < handle
% Natural Evolution Strategy with optimal baseline.
% It supports Importance Sampling (IS).
%
% =========================================================================
% REFERENCE
% D Wierstra, T Schaul, T Glasmachers, Y Sun, J Peters, J Schmidhuber
% Natural Evolution Strategy (2014)
    
    properties
        epsilon % learning rate parameter
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = NES_Solver(epsilon)
            obj.epsilon = epsilon;
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, div] = step(obj, J, Actions, policy, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights

            [nat_grad, stepsize] = NESbase(obj, policy, J, Actions, W);
            div = norm(nat_grad);

            policy = policy.update(policy.theta + nat_grad * stepsize);
        end

        %% CORE
        function [nat_grad, stepsize] = NESbase(obj, policy, J, Actions, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            
            dlogPidtheta = policy.dlogPidtheta(Actions);
            J = permute(J,[3 2 1]);
            
            den = sum( bsxfun(@times, dlogPidtheta.^2, W.^2), 2 );
            num = sum( bsxfun(@times, dlogPidtheta.^2, bsxfun(@times, J, W.^2)), 2 );
            b = bsxfun(@times, num, 1./den);
            b(isnan(b)) = 0;
            diff = bsxfun(@times, bsxfun(@minus,J,b), W);
            grad = permute( sum(bsxfun(@times, dlogPidtheta, diff), 2), [1 3 2] );

%             N = sum(W); % lower variance
            N = length(W); % unbiased
            
            grad = grad / N;

            % If we can compute the FIM in closed form, we use it
            if ismethod(policy,'fisher')
                F = policy.fisher;
            else
                F = dlogPidtheta * bsxfun(@times, dlogPidtheta, W)';
                F = F / N;
            end
            
            % If we can compute the FIM inverse in closed form, we use it
            if ismethod(policy,'inverseFisher')
                invF = policy.inverseFisher;
                nat_grad = invF * grad;
            elseif rank(F) == size(F,1)
                nat_grad = F \ grad;
            else
%                 warning('Fisher matrix is lower rank (%d instead of %d).', rank(F), size(F,1));
                nat_grad = pinv(F) * grad;
            end
            
            lambda = sqrt(diag(grad' * nat_grad) / (4 * obj.epsilon))';
            lambda = max(lambda,1e-8); % to avoid numerical problems
            stepsize = 1 ./ (2 * lambda);
        end
        
    end
    
end
