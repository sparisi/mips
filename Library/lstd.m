function [A, b] = lstd(data, gamma, lambda, prob_ratio, A, b)
% LSTD(lambda) for approximate policy evaluation.
% Do not pass PROB_RATIO if data is on-policy.
% Truncate PROB_RATIO = min(1,PROB_RATIO) to use Retrace.

% =========================================================================
% REFERENCE
% J A Boyan
% Least-Squares Temporal Difference Learning (1998)

phi = [data.phiV];
phiN = [data.phiV_nexts];
r = [data.r];
t = [data.t];

if nargin < 4 || isempty(prob_ratio), prob_ratio = ones(size(r)); end
if nargin < 5 || isempty(A) || isempty(b), A = 0; b = 0; end

for k = 1 : size(r,2)
    if t(k) == 1, z = phi(:,k); end
    A = A + z * (phi(:,k) - gamma * phiN(:,k))';
    b = b + z * r(:,k);
    z = prob_ratio(k) * gamma * lambda * z + phiN(:,k);
end
