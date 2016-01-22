function [nat_grad, stepsize] = NESbase(pol_high, J, Theta, lrate)
% Natural Evolution Strategy with optimal baseline.
% NAT_GRAD is a [D x R] matrix, where D is the length of the gradient and R
% is the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% D Wierstra, T Schaul, T Glasmachers, Y Sun, J Peters, J Schmidhuber 
% Natural Evolution Strategy (2014)

dlogPidtheta = pol_high.dlogPidtheta(Theta);
J = permute(J,[3 2 1]);

den = sum(dlogPidtheta.^2,2);
num = sum( bsxfun(@times, dlogPidtheta.^2, J), 2);
b = bsxfun(@times, num, 1./den);
b(isnan(b)) = 0;
diff = bsxfun(@minus,J,b);
grad = permute( mean(bsxfun(@times, dlogPidtheta, diff), 2), [1 3 2]);

% If we can compute the FIM in closed form, we use it
if ismethod(pol_high,'fisher')
    F = pol_high.fisher;
else
    F = dlogPidtheta * dlogPidtheta' / size(J,2);
end

% If we can compute the FIM inverse in closed form, we use it
if ismethod(pol_high,'inverseFisher')
    invF = pol_high.inverseFisher;
    nat_grad = invF * grad;
elseif rank(F) == size(F,1)
    nat_grad = F \ grad;
else
%     warning('Fisher matrix is lower rank (%d instead of %d).', rank(F), size(F,1));
    nat_grad = pinv(F) * grad;
end

if nargin == 4
    lambda = sqrt(diag(grad' * nat_grad) / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 ./ (2 * lambda);
end

end