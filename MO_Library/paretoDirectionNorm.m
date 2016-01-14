function minNorm = paretoDirectionNorm(D_theta_J)
% PARETODIRECTIONNORM Given the Jacobian matrix D_THETA_J (each row i is 
% the derivative of the objective function J_i wrt theta), the function 
% finds the norm of the min-norm Pareto-ascent direction. 
% The general problem requires to solve a quadratic programming problem.
% This function solves a very special case, i.e., when the number of
% objectives J is less then four and equal to the number of parameters theta.
%
% =========================================================================
% REFERENCE
% J A Desideri
% Multiple-Gradient Descent Algorithm (MGDA) (2012)

dim_J = size(D_theta_J,1);

if dim_J == 2

    g1 = transpose(D_theta_J(1,:));
    g2 = transpose(D_theta_J(2,:));
    g1 = g1/norm(g1);
    g2 = g2/norm(g2);
    l = 0.5;
    dir = l*g1+(1-l)*g2;
    minNorm = norm(dir);

elseif dim_J == 3

    g1 = transpose(D_theta_J(1,:));
    g2 = transpose(D_theta_J(2,:));
    g3 = transpose(D_theta_J(3,:));
    g1 = g1/norm(g1);
    g2 = g2/norm(g2);
    g3 = g3/norm(g3);
    n = cross((g2-g1),(g3-g1));
    d = abs( transpose(n) * g1 );
    minNorm = d / norm(n);

else
    
    error('Symbolic Pareto-ascent norm with more than 3 objectives not yet implemented.')
    % See Wedge product and exterior algebra

end

end