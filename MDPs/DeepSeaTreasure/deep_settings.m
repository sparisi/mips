function [ n_obj, policy, episodes, steps, gamma ] = deep_settings

mdp_vars = deep_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
nactions = length(mdp_vars.action_list);

bfs = @deep_basis_poly;
policy = gibbs(bfs, zeros(bfs()*(nactions-1),1), mdp_vars.action_list);
% policy = gibbs_allpref(bfs, zeros(bfs()*nactions,1), mdp_vars.action_list);

%%% Evaluation
episodes = 500;
steps = 50;

%%% Learning
episodes = 250;
steps = 50;

end
