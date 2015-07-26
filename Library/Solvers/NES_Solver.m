%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: D Wierstra, T Schaul, T Glasmachers, Y Sun, J Peters
% J Schmidhuber (2014)
% Natural Evolution Strategy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef NES_Solver < handle
    
    % Natural Evolution Strategies
    
    properties(GetAccess = 'public', SetAccess = 'private')
        lrate;
        N;       % number of samples
        policy;  % distribution for sampling the episodes
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = NES_Solver(lrate, N, policy)
            obj.lrate = lrate;
            obj.N = N;
            obj.policy = policy;
        end
        
        %% SETTER
        function obj = setPolicy(obj, policy)
            obj.policy = policy;
        end
        
        %% CORE
        function div = step(obj, J, Theta)
            [nat_grad, div] = NESbase(obj, J, Theta);
            update(obj, nat_grad);
        end
        
        function [nat_grad, div] = NESbase(obj, J, Theta)
            n_episodes = length(J);
            
            num = 0;
            den = 0;
            dlogPidtheta = zeros(obj.policy.dlogPidtheta,n_episodes);
            
            % Compute optimal baseline
            for k = 1 : n_episodes
                
                dlogPidtheta(:,k) = obj.policy.dlogPidtheta(Theta(:,k));
                
                num = num + dlogPidtheta(:,k).^2 * J(k);
                den = den + dlogPidtheta(:,k).^2;
                
            end
            
            b = num ./ den;
            b(isnan(b)) = 0;
            % b = mean_J;
            
            % Estimate gradient and Fisher information matrix
            grad = 0;
            F = 0;
            for k = 1 : n_episodes
                grad = grad + dlogPidtheta(:,k) .* (J(k) - b);
                F = F + dlogPidtheta(:,k) * dlogPidtheta(:,k)';
            end
            grad = grad / n_episodes;
            F = F / n_episodes;
            
            % If we can compute the FIM in closed form, use it
            if ismethod(obj.policy,'fisher')
                F = obj.policy.fisher;
            end
            
            % If we can compute the FIM inverse in closed form, use it
            if ismethod(obj.policy,'inverseFisher')
                invF = obj.policy.inverseFisher;
                nat_grad = invF * grad;
            elseif rank(F) == size(F,1)
                nat_grad = F \ grad;
            else
%                 warning('F is lower rank (rank = %d)!!! Should be %d', rank(F), size(F,1));
                nat_grad = pinv(F) * grad;
            end
            lambda = sqrt(grad' * nat_grad / (4 * obj.lrate));
            lambda = max(lambda,1e-8); % to avoid numerical problems
            stepsize = 1 / (2 * lambda);
            
            div = norm(nat_grad);
            nat_grad = nat_grad * stepsize;
        end
        
        function update(obj, gradient)
            obj.policy = obj.policy.update(gradient);
        end
        
    end
    
end
