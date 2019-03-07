function [nat_grad, stepsize] = NPG_C(policy, data, gamma, lrate)
% As NPG, but with compatible function approximation. 
% See REINFORCE_C.

R = mc_ret(data,gamma);
R = (R - mean(R)) / std(R); % standardize data
dlogpi = policy.dlogPidtheta([data.s],[data.a]);

options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);
w = fminunc(@(w)mse_linear(w,dlogpi,R), zeros(size(dlogpi,1),1), options);
    
if nargin == 4
    R = w'*dlogpi;
    grad = dlogpi * R' / size(R,2);
    F = dlogpi * dlogpi' / size(R,2);
    
    if rank(F) == size(F,1)
        nat_grad = F \ grad;
    else
%         warning('Fisher matrix is lower rank (%d instead of %d).', rank(F), size(F,1));
        nat_grad = pinv(F) * grad;
    end
    
    lambda = sqrt(diag(grad' * nat_grad) / (4 * lrate))';
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 ./ (2 * lambda);
end

nat_grad = w;