function phi = bicycledrive_basis_poly(state)
% REFERENCE
% M G Lagoudakis, R Parr
% Least-Squares Policy Iteration (2003)

if nargin == 0
    phi = 19;
    return
end

theta       = state(1,:);
theta_dot   = state(1,:);
omega       = state(3,:);
omega_dot   = state(4,:);
psi         = state(5,:);

idx = psi > 0;
psi_hat = (pi - psi) .* ones(1,size(state,2));
psi_hat(idx) = -pi - psi(idx);

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
    omega.^2 .* theta
    psi
    psi.^2
    psi .* theta
    psi_hat
    psi_hat.^2
    psi_hat .* theta];
