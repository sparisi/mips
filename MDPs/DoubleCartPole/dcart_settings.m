function [ n_obj, policy, episodes, steps, gamma ] = dcart_settings

mdp_vars = dcart_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
nactions = length(mdp_vars.action_list);

bfs = @dcart_basis_poly;
policy = gibbs(bfs, zeros(bfs()*(nactions-1),1), mdp_vars.action_list);

episodes = 100;
steps = 1000;

end
