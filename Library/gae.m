function A = gae(data, V, gamma, lambda)
% Computes generalized advantage estimates from potentially off-policy data.
%
% =========================================================================
% REFERENCE
% J Schulman, P Moritz, S Levine, M Jordan, P Abbeel
% High-Dimensional Continuous Control Using Generalized Advantage Estimation
% ICLR (2017)

A = zeros(size(V));
r = [data.r];
done = [data.endsim];
n = size(V,2);

for k = n : -1 : 1
    if done(k)
        A(k) = r(k) - V(k);
    else
        A(k) = r(k) + gamma*V(k+1) - V(k) + gamma*lambda*A(k+1);
    end
end
