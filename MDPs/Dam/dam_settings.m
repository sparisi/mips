function [ n_obj, policy, episodes, steps, gamma ] = dam_settings

mdp_vars = dam_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
dim = mdp_vars.nvar_action;

bfs = @dam_basis_rbf;
k0 = [50, -50, 0, 0, 50];
% k0 = zeros(1,bfs());
% policy = gaussian_fixedvar(bfs, dim, k0, 0.1);
% policy = gaussian_linear(bfs, dim, k0, 20);
policy = gaussian_diag_linear(bfs, dim, k0, 20);
% policy = gaussian_logistic_linear(bfs, dim, k0, 1, 50);

%%% Evaluation
episodes = 1000;
steps = 100;

%%% Learning
episodes = 150;
steps = 50;

end
