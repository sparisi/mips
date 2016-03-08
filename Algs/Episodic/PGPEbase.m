function [grad, stepsize] = PGPEbase (pol_high, J, Theta, lrate, W)
% Policy Gradient with Parameter-based Exporation and optimal baseline.
% It supports Importance Sampling (IS).
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% F Sehnke, C Osendorfer, T Rueckstiess, A Graves, J Peters, J Schmidhuber 
% Parameter-exploring Policy Gradients (2010)
%
% T Zhao, H Hachiya, M Sugiyama
% Analysis and Improvement of Policy Gradient Estimation (2011)
%
% T Zhao, H Hachiya, V Tangkaratt, J Morimoto, M Sugiyama
% Effcient Sample Reuse in Policy Gradients with Parameter-based 
% Exploration (2013)
%
% J Tang, P Abbeel
% On a Connection between Importance Sampling and the Likelihood Ratio 
% Policy Gradient (2010)

if nargin < 5, W = ones(1, size(J,2)); end % IS weights

dlogPidtheta = pol_high.dlogPidtheta(Theta);
J = permute(J,[3 2 1]);

den = sum( bsxfun(@times, dlogPidtheta.^2, W.^2), 2 );
num = sum( bsxfun(@times, dlogPidtheta.^2, bsxfun(@times, J, W.^2)), 2 );
b = bsxfun(@times, num, 1./den);
b(isnan(b)) = 0;
diff = bsxfun(@times, bsxfun(@minus,J,b), W);
grad = permute( sum(bsxfun(@times, dlogPidtheta, diff), 2), [1 3 2] );

% grad = grad / length(W); % unbiased
grad = grad / sum(W); % lower variance

if nargin > 3
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end

end
