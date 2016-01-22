function [grad, stepsize] = PGPEbase (pol_high, J, Theta, lrate)
% Policy Gradient with Parameter-based Exporation with optimal baseline.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% F Sehnke, C Osendorfer, T Rueckstiess, A Graves, J Peters, J Schmidhuber 
% Parameter-exploring Policy Gradients (2010)

dlogPidtheta = pol_high.dlogPidtheta(Theta);
J = permute(J,[3 2 1]);

den = sum(dlogPidtheta.^2,2);
num = sum( bsxfun(@times, dlogPidtheta.^2, J), 2);
b = bsxfun(@times, num, 1./den);
b(isnan(b)) = 0;
diff = bsxfun(@minus,J,b);
grad = permute( mean(bsxfun(@times, dlogPidtheta, diff), 2), [1 3 2]);

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end

end
