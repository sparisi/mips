function [ n_obj, policy, episodes, steps, gamma ] = mcs_settings

mdp_vars = mcs_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

bfs = @mcs_basis_poly;
policy = gibbs(bfs, zeros(bfs(),1), mdp_vars.action_list);

episodes = 100;
steps = 1000;

end
