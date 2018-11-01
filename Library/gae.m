function A = gae(data, V, gamma, lambda)
% Computes generalized advantage estimates from potentially off-policy data.
%
% =========================================================================
% REFERENCE
% J Schulman, P Moritz, S Levine, M Jordan, P Abbeel
% High-Dimensional Continuous Control Using Generalized Advantage Estimation
% ICLR (2017)

r = [data.r];
t = [data.t];
t(end+1) = 1;
A = zeros(size(V));

for k = size(V,2) : -1 : 1
    if t(k+1) == 1 % Next state is a new episode init state
        A(k) = r(k) - V(k);
    else
        A(k) = r(k) + gamma*V(k+1) - V(k) + gamma*lambda*A(k+1);
    end
end
