function [ n_obj, policy, episodes, steps, gamma ] = resource_settings

mdp_vars = resource_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

bfs = @resource_basis_poly;
policy = gibbs(bfs, zeros(bfs(),1), mdp_vars.action_list);

episodes = 150;
steps = 40;

end

