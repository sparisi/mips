function phi = bicyclebalance_basis_poly(state)
% REFERENCE
% M G Lagoudakis, R Parr
% Least-Squares Policy Iteration (2003)

if nargin == 0
    phi = 13;
    return
end

theta     = state(1,:);
theta_dot = state(2,:);
omega     = state(3,:);
omega_dot = state(4,:);

phi = [omega
    omega_dot
    omega.^2
    omega_dot.^2
    omega .* omega_dot
    theta
    theta_dot
    theta.^2
    theta_dot.^2
    theta .* theta_dot
    omega .* theta
    omega .* theta.^2
    omega.^2 .* theta];
