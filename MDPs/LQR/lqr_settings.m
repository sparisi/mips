function [ n_obj, policy, episodes, steps, gamma ] = lqr_settings

mdp_vars = lqr_mdpvariables();
dim = mdp_vars.nvar_action;
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

bfs = @lqr_basis_poly;
A0 = -0.5 * eye(dim);
Sigma0 = 1 * eye(dim);
a0 = zeros(dim,1);
w0 = ones(dim,1);
tau = ones(dim,1);
% policy = gaussian_chol_linear(bfs, dim, A0, Sigma0);
% policy = gaussian_diag_linear(bfs, dim, A0, Sigma0);
% policy = gaussian_linear(bfs, dim, A0, Sigma0);
% policy = gaussian_linear_full(bfs, dim, a0, A0, Sigma0);
% policy = gaussian_logistic_linear(bfs, dim, A0, w0, tau);
% policy = gaussian_fixedvar_linear(bfs, dim, A0, Sigma0);
policy = gaussian_fixedvar_linear_diagmean(bfs, dim, diag(A0), Sigma0);

%%% Evaluation
episodes = 150;
steps = 50;

%%% Learning
episodes = 10;
steps = 30;

end
