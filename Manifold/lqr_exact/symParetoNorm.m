function minNorm = symParetoNorm(D_theta_J)
% SYMPARETONORM Computes the symbolic minimum-norm Pareto-ascent direction
% given the symbolic Jacobian D_theta_J.

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