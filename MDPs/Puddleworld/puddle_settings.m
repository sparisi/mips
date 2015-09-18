function [ n_obj, policy, episodes, steps, gamma ] = puddle_settings

mdp_vars = puddle_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
nactions = length(mdp_vars.action_list);

bfs = @puddle_basis_rbf;
policy = gibbs(bfs, zeros(bfs()*(nactions-1),1), mdp_vars.action_list);

episodes = 15;
steps = 10;

end
