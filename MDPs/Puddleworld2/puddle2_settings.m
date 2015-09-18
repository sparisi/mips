function [ n_obj, policy, episodes, steps, gamma ] = puddle2_settings

mdp_vars = puddle2_mdpvariables();
n_obj = mdp_vars.nvar_reward;
n_act = mdp_vars.nvar_action;
gamma = mdp_vars.gamma;

bfs = @puddle2_basis_rbf;
policy = gaussian_diag_linear(bfs, n_act, zeros(n_act,bfs()), [10; 10]);

episodes = 15;
steps = 10;

end
