function [grad, stepsize] = PGPEbase (pol_high, J, Theta, lrate)
% Policy Gradient with Parameter-based Exporation with optimal baseline.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% F Sehnke, C Osendorfer, T Rueckstiess, A Graves, J Peters, J Schmidhuber 
% Parameter-exploring Policy Gradients (2010)

[n_objectives, ~] = size(J);
dlogPidtheta = pol_high.dlogPidtheta(Theta);
grad = zeros(pol_high.dparams, n_objectives);

for i = 1 : n_objectives
    den = sum(dlogPidtheta.^2,2);
    num = sum(bsxfun(@times,dlogPidtheta.^2,J(i,:)),2);
    b = num ./ den;
    b(isnan(b)) = 0;

    grad(:,i) = mean(dlogPidtheta .* bsxfun(@minus,J(i,:),b),2);
end

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end

end
