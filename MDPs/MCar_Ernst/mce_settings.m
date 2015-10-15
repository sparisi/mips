function [ n_obj, policy, episodes, steps, gamma ] = mce_settings

mdp_vars = mce_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
nactions = length(mdp_vars.action_list);

bfs = @mce_basis_rbf;
policy = gibbs(bfs, zeros(bfs()*(nactions-1),1), mdp_vars.action_list);

episodes = 500;
steps = 150;

end
