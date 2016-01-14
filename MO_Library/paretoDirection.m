function [dir, lambda] = paretoDirection(jacobian)
% PARETODIRECTION Computes the minimum-norm Pareto-ascent direction.
%
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

N_obj = size(jacobian,2);
options = optimset('Display', 'off',...
                   'Algorithm', 'interior-point-convex');
lambda = quadprog(jacobian'*jacobian, ...
    zeros(N_obj,1), [], [], ones(1,N_obj), 1, zeros(1,N_obj), ...
    [], ones(N_obj,1)/N_obj, options);
dir = jacobian * lambda;

end