classdef PFA_Solver < handle
% Pareto-Following Algorithm.
%
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

    properties
        lrate_single % lrate during single-objective optimization phase
        lrate_step   % lrate during optimization step
        lrate_corr   % lrate during correction step
        gradient     % handle for the algorithm for computing the gradient
        gamma        % discount factor
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = PFA_Solver(lrate_single, lrate_step, lrate_corr, gamma, alg)
            obj.lrate_single = lrate_single;
            obj.lrate_step = lrate_step;
            obj.lrate_corr = lrate_corr;
            obj.gamma = gamma;
            obj.gradient = alg;
        end
        
        %% OPTIMIZATION
        function [policy, gnorm] = optimization_step(obj, data, policy, objective)
            [grad, stepsize] = obj.gradient(policy,data,obj.gamma,obj.lrate_single);
            grad = grad(:,objective);
            stepsize = stepsize(objective);
            gnorm = norm(grad);
            policy = policy.update(policy.theta + grad * stepsize);
        end
        
        %% CORRECTION
        function [policy, gnorm] = correction_step(obj, data, policy)
            grads = obj.gradient(policy,data,obj.gamma,obj.lrate_single);
            normgrads = matrixnorms(grads,2);
            grads = bsxfun(@times,grads,1./normgrads); % Always normalize during the correction
            pareto_dir = paretoDirection(grads); % Minimal-norm Pareto-ascent direction
            gnorm = norm(pareto_dir);
            policy = policy.update(policy.theta + obj.lrate_corr * pareto_dir); % Move towards the frontier
        end

    end
    
end
