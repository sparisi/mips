function [ind_d] = metric_hv_relu_d(J, P, beta1, beta2, normalize)
% Separate function for its derivative, needed because of function handles.

if nargin == 4, normalize = true; end
assert(beta1 >= 0 && beta1 <= 1, 'BETA1 must be within 0 and 1.')
assert(beta2 >= 0 && beta2 <= 1, 'BETA2 must be within 0 and 1.')

% Get sizes, antiutopia, utopia
[n_f, d] = size(J);
[n_p, ~] = size(P);
AU = min(P, [], 1);
U = max(P, [], 1);

% Normalize
if normalize
    P = bsxfun(@times, bsxfun(@minus, P, AU), 1 ./ (U - AU));
    J = bsxfun(@times, bsxfun(@minus, J, AU), 1 ./ (U - AU));
    AU_orig = AU;
    U_orig = U;
    AU = min(P, [], 1);
    U = max(P, [], 1);
end

% Transform for bsxfun
F_T = permute(J, [3,2,1]);

% Get Pareto front idx
[ND, ~, tmp] = pareto(J);
i_nd = false(n_f,1);
i_nd(tmp) = true;

% Reward
d_fp = bsxfun(@plus, -P, F_T);
i_fp = d_fp > 0;
reward = d_fp .* i_fp;
reward_d = 2 * reward;
reward = reward.^2;
reward = sum(reward, 2);
reward = sum(reward, 1);
reward_d = sum(reward_d, 1);
reward = squeeze(reward);
reward_d = squeeze(reward_d);

% Penalty
d_pf = bsxfun(@plus, P, -F_T);
i_pf = d_pf > 0;
penalty = d_pf .* i_pf;
penalty_d = - 2 * penalty;
penalty = penalty.^2;
penalty = sum(penalty, 2);
penalty = sum(penalty, 1);
penalty_d = sum(penalty_d, 1);
penalty = squeeze(penalty);
penalty_d = squeeze(penalty_d);

% Penalty antiutopia
d_auf = bsxfun(@plus, AU, -F_T);
i_auf = d_auf > 0;
penalty_au = d_auf .* i_auf;
penalty_au_d = - 2 * penalty_au;
penalty_au = penalty_au.^2;
penalty_au = sum(penalty_au, 2);
penalty_au = sum(penalty_au, 1);
penalty_au_d = sum(penalty_au_d, 1);
penalty_au = squeeze(penalty_au);
penalty_au_d = squeeze(penalty_au_d);

penalty_au = penalty_au * n_f * n_p * d;
penalty_au_d = penalty_au_d * n_f * n_p * d;

% Transpose derivative
reward_d = reward_d';
penalty_d = penalty_d';
penalty_au_d = penalty_au_d';

% Sum components
ind = zeros(n_f,1);
ind(i_nd) = ind(i_nd) + ...
    reward(i_nd) - beta1 * penalty(i_nd) - penalty_au(i_nd);
ind(~i_nd) = ind(~i_nd) + ...
    beta2 * reward(~i_nd) - penalty(~i_nd) - penalty_au(~i_nd);

ind_d = zeros(n_f,d);
ind_d(i_nd,:) = ind_d(i_nd,:) + ...
    reward_d(i_nd,:) - beta1 * penalty_d(i_nd,:) - penalty_au_d(i_nd,:);
ind_d(~i_nd,:) = ind_d(~i_nd,:) + ...
    beta2 * reward_d(~i_nd,:) - penalty_d(~i_nd,:) - penalty_au_d(~i_nd,:);

% Derivative of normalization
if normalize
    ind_d = bsxfun(@times, ind_d, 1./ (U_orig - AU_orig));
end
