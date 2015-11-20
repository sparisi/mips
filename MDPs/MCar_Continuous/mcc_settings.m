function [ n_obj, policy, episodes, steps, gamma ] = mcc_settings

mdp_vars = mce_mdpvariables();
n_obj = mdp_vars.nvar_reward;
n_act = mdp_vars.nvar_action;
gamma = mdp_vars.gamma;

bfs = @mcc_basis_rbf;
policy = gaussian_diag_linear(bfs, n_act, zeros(n_act,bfs()), 10);

episodes = 100;
steps = 100;

end
