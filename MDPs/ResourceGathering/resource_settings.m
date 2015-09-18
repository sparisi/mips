function [ n_obj, policy, episodes, steps, gamma ] = resource_settings

mdp_vars = resource_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
nactions = length(mdp_vars.action_list);

bfs = @resource_basis_poly;
policy = gibbs(bfs, zeros(bfs()*(nactions-1),1), mdp_vars.action_list);
% policy = gibbs_allpref(bfs, zeros(bfs()*nactions,1), mdp_vars.action_list);

episodes = 150;
steps = 40;

end

