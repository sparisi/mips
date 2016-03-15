classdef PFA_Solver < handle
% Pareto-Following Algorithm.
%
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

    properties
        lrate_step   % lrate during optimization step
        lrate_corr   % lrate during correction step
        gradient     % handle for the algorithm for computing the gradient
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = PFA_Solver(lrate_step, lrate_corr, alg)
            obj.lrate_step = lrate_step;
            obj.lrate_corr = lrate_corr;
            obj.gradient = alg;
        end
        
        %% OPTIMIZATION
        function [policy, gnorm] = optimization_step(obj, data, policy, objective)
            [grad, stepsize] = obj.gradient(policy, data, obj.lrate_step);
            grad = grad(:,objective);
            stepsize = stepsize(objective);
            gnorm = norm(grad);
            policy = policy.update(policy.theta + grad * stepsize);
        end
        
        %% CORRECTION
        function [policy, gnorm] = correction_step(obj, data, policy)
            grads = obj.gradient(policy, data, obj.lrate_corr);
            normgrads = max(matrixnorms(grads,2),1e-8);
            grads = bsxfun(@times,grads,1./normgrads); % Always normalize during the correction
            pareto_dir = paretoDirection(grads); % Minimal-norm Pareto-ascent direction
            gnorm = norm(pareto_dir);
            policy = policy.update(policy.theta + obj.lrate_corr * pareto_dir); % Move towards the frontier
        end

    end
    
end
