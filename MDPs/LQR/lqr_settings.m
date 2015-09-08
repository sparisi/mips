function [ n_obj, policy, episodes, steps, gamma ] = lqr_settings

mdp_vars = lqr_mdpvariables();
dim = mdp_vars.nvar_action;
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

bfs = @lqr_basis_poly;
k0 = -0.5 * eye(dim);
s0 = 1 * eye(dim);
offset0 = zeros(dim,1);
tau = 1.3 * ones(size(diag(s0)));
% policy = gaussian_logistic_linear(bfs, dim, k0, diag(s0), tau);
% policy = gaussian_linear(bfs, dim, k0, s0);
% policy = gaussian_diag_linear(bfs, dim, k0, diag(s0));
% policy = gaussian_linear_full(bfs, dim, offset0, k0, (s0));
% policy = gaussian_chol_linear(bfs, dim, k0, chol(s0));
policy = gaussian_fixedvar(bfs, dim, k0, s0);

%%% Evaluation
episodes = 150;
steps = 50;

%%% Learning
episodes = 50;
steps = 50;

end
