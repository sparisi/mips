%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: F Sehnke, C Osendorfer, T Rueckstiess, A Graves, J Peters, 
% Juergen Schmidhuber (2010)
% Parameter-exploring Policy Gradients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad, stepsize] = PGPEbase (pol_high, J, Theta, lrate)

n_episodes = length(J);

num = 0;
den = 0;
dlogPidtheta = zeros(pol_high.dlogPidtheta,n_episodes);

% Compute optimal baseline
for k = 1 : n_episodes
    
    dlogPidtheta(:,k) = pol_high.dlogPidtheta(Theta(:,k));
    
    num = num + dlogPidtheta(:,k).^2 * J(k);
    den = den + dlogPidtheta(:,k).^2;
    
end

b = num ./ den;
b(isnan(b)) = 0;
% b = mean(J);

% Estimate gradient
grad = 0;
for k = 1 : n_episodes
    grad = grad + dlogPidtheta(:,k) .* (J(k) - b);
end
grad = grad / n_episodes;

if nargin >= 4
    T = eye(length(grad));
    lambda = sqrt(grad' * T * grad / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end

end
