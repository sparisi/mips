function A = retrace(data, V, gamma, lambda, prob_ratio)
% Computes generalized advantage estimates from off-policy data.
% Data have to be ordered by episode: data.r must have first all samples 
% from the first episode, then all samples from the second, and so on.
% So you cannot use samples collected with COLLECT_SAMPLES2.
%
% =========================================================================
% REFERENCE
% R Munos, T Stepleton, Anna Harutyunyan, M G Bellemare
% Safe and efficient off-policy reinforcement learning
% NIPS (2016)

prob_ratio = min(1, prob_ratio);
r = [data.r];
t = [data.t];
t(end+1) = 1;
A = zeros(size(V));

for k = size(V,2) : -1 : 1
    if t(k+1) == 1 % Next state is a new episode init state
        A(k) = prob_ratio(k) * (r(k) - V(k));
    else
        A(k) = prob_ratio(k) * (r(k) + gamma*V(k+1) - V(k) + gamma*lambda*A(k+1));
    end
end
