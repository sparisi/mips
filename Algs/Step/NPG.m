function [nat_grad, stepsize] = NPG(policy, data, gamma, lrate)
% Basic version of natural policy gradient. REINFORCE gradient with baseline
% mean(R) is multiplied by the inverse of the Fisher information matrix.

R = mc_ret(data,gamma);
R = R - mean(R,2);
dlogpi = policy.dlogPidtheta([data.s],[data.a]);
grad = dlogpi * R' / size(R,2);

F = dlogpi * dlogpi' / size(R,2);

if rank(F) == size(F,1)
    nat_grad = F \ grad;
else
%     warning('Fisher matrix is lower rank (%d instead of %d).', rank(F), size(F,1));
    nat_grad = pinv(F) * grad;
end

if nargin == 4
    lambda = sqrt(diag(grad' * nat_grad) / (4 * lrate))';
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 ./ (2 * lambda);
end
