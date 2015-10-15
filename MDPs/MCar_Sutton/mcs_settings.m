function [ n_obj, policy, episodes, steps, gamma ] = mcs_settings

mdp_vars = mcs_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
nactions = length(mdp_vars.action_list);

bfs = @mcs_basis_poly;
policy = gibbs(bfs, zeros(bfs()*(nactions-1),1), mdp_vars.action_list);

episodes = 10;
steps = 100;

end
